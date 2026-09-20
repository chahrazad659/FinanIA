"""Classeur de réponses → User choisi → profil automatique → RAG + Ollama."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
import streamlit as st
from finrag.ingest import DATA, load_corpus
from finrag.profile import load_schema, load_excel_dataset, build_profile, build_query
from finrag.ollama_client import OllamaClient, OllamaError
from finrag.retrieval import Retriever, RetrievalError
from finrag.generation import generate_recommendations, GenerationError


@st.cache_data
def corpus():
    return load_corpus()


@st.cache_data(show_spinner=False, max_entries=2)
def workbook_profiles(content: bytes):
    """Cache mémoire uniquement ; aucun fichier importé enregistré sur disque."""
    return load_excel_dataset(BytesIO(content))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def display_value(value):
    if value is None or value == "" or value == []:
        return "Non renseigné"
    return " ; ".join(value) if isinstance(value, list) else str(value)


def main():
    st.set_page_config(page_title="Financement · Recommandations par User", page_icon="📚", layout="wide")
    schema = load_schema()
    chunks, catalog, manifest = corpus()
    with st.sidebar:
        st.title("Financement")
        st.caption("Excel → User → Profil → RAG → LLM")
        mode = st.radio("Analyse", ["RAG + Ollama local", "Extraits sans LLM"], key="mode")
        with st.expander("Réglages Ollama"):
            host = st.text_input("Adresse locale", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
            chat_model = st.text_input("Modèle de conversation", os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct-q4_K_M"))
            embed_model = st.text_input("Modèle d'embeddings", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"))
            if st.button("Vérifier Ollama"):
                try:
                    st.write(OllamaClient(host, chat_model, embed_model, timeout=5).tags())
                    st.success("Ollama est joignable.")
                except (OllamaError, ValueError) as exc:
                    st.error(str(exc))
        semantic = st.checkbox("Recherche hybride avec embeddings", value=(DATA / "index" / "embeddings.json").exists(),
                               help="Construisez l'index une fois avant d'activer cette option. Sinon : recherche BM25.")
        count = st.slider("Nombre maximum de programmes", 1, 5, 3)
        if st.button("Construire l'index sémantique", width="stretch"):
            try:
                with st.spinner("Vectorisation locale du guide…"):
                    client = OllamaClient(host, chat_model, embed_model, timeout=600)
                    size = Retriever(chunks, DATA / "index", client).build_embeddings()
                st.success(f"Index prêt : {size} passages.")
            except (OllamaError, RetrievalError, ValueError) as exc:
                st.error(str(exc))
        st.divider()
        st.caption("Guide de décembre 2025. Disponibilité actuelle des programmes à confirmer.")
        st.caption("Les imports restent en mémoire. Aucun envoi vers une API cloud.")

    st.title("Recommandations de financement par User")
    st.write("Chargez le fichier contenant les réponses, choisissez un User et obtenez les programmes proposés à partir de son profil. Aucun questionnaire à remplir.")
    choose, result_tab, library, method = st.tabs(["1 · Choisir un User", "2 · Programmes proposés", "3 · Sources", "4 · Méthode"])
    with choose:
        uploaded = st.file_uploader("Fichier Excel contenant les réponses", type=["xlsx"],
                                    help="Colonnes : User, Q1…Q134. Feuille : Reponses_synthetiques.")
        source_name = uploaded.name if uploaded is not None else "Questionnaire_10000_users_synthetiques.xlsx"
        file_content = uploaded.getvalue() if uploaded is not None else (DATA / "raw" / source_name).read_bytes()
        if uploaded is None:
            st.info("Le classeur fourni au début est déjà chargé. Choisissez directement un de ses 10 000 Users, ou importez un autre fichier de même structure.")
        try:
            with st.spinner("Lecture des réponses Excel… La première lecture peut prendre quelques instants."):
                records = workbook_profiles(file_content)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            st.error(f"Lecture impossible : {exc}")
            return
        source_id = hashlib.sha256(file_content).hexdigest()
        a, b, c = st.columns(3)
        a.metric("Users dans le fichier", len(records))
        b.metric("Réponses possibles par User", len(schema["questions"]))
        c.metric("Dispositifs dans le guide", len(catalog))
        selected_user = st.selectbox("Choisir le User à analyser", list(records), key=f"user_{source_id}",
                                     help="Tapez un identifiant, par exemple User 500, pour le retrouver.")
        st.caption(f"Fichier : {source_name} · Les réponses de {selected_user} sont lues automatiquement.")
        profile = build_profile(records[selected_user], schema)
        selection = (source_id, selected_user)
        if st.session_state.get("selection") != selection:
            st.session_state.selection = selection
            st.session_state.pop("result", None)
            st.session_state.pop("hits", None)
        st.session_state.selected_user = selected_user
        st.session_state.profile = profile
        answers = profile["answers"]
        cards = st.columns(3)
        cards[0].metric("Secteur", display_value(answers.get("Q19")))
        cards[1].metric("Montant recherché (DH)", display_value(answers.get("Q32")))
        cards[2].metric("Région", display_value(answers.get("Q11")))
        st.write("**Activité :** " + display_value(answers.get("Q17")))
        st.write("**Objectif :** " + display_value(answers.get("Q33")))
        if profile["warnings"]:
            with st.expander(f"{len(profile['warnings'])} points à vérifier dans les réponses"):
                for warning in profile["warnings"]:
                    st.warning(warning)
        with st.expander(f"Voir les 134 réponses de {selected_user}"):
            st.dataframe([{"Question": q["id"], "Libellé": q["label"], "Réponse lue": display_value(answers.get(q["id"]))} for q in schema["questions"]], hide_index=True, width="stretch")
        st.caption("Le fichier fourni contient des profils synthétiques pour les essais, sans programme cible de référence.")
        run_from_user = st.button(f"Recommander les programmes pour {selected_user}", key="recommend_user", type="primary", width="stretch")

    with result_tab:
        st.subheader(f"Programmes proposés pour {selected_user}")
        st.caption("Le profil est construit automatiquement depuis les réponses de la ligne sélectionnée.")
        run_from_result = st.button("Lancer l'analyse du User sélectionné", key="recommend_result", type="primary")
        if run_from_user or run_from_result:
            st.session_state.pop("result", None)
            st.session_state.pop("hits", None)
            try:
                with st.spinner(f"Recherche et analyse du profil de {selected_user}…"):
                    query = build_query(profile, schema)
                    client = OllamaClient(host, chat_model, embed_model, timeout=600)
                    hits = Retriever(chunks, DATA / "index", client if semantic else None).search(query, k=8, semantic=semantic)
                    llm_profile = {"reponses": profile["labeled_answers"], "incoherences": profile["warnings"]}
                    result = generate_recommendations(llm_profile, query, hits, client if mode == "RAG + Ollama local" else None, count)
                st.session_state.result = result
                st.session_state.hits = hits
                st.session_state.result_query = query
                st.session_state.result_time = datetime.now(timezone.utc).isoformat()
            except (OllamaError, RetrievalError, GenerationError, ValueError, OSError) as exc:
                st.error(str(exc))
        if "result" not in st.session_state:
            st.info("Choisissez un User dans le premier onglet, puis lancez son analyse.")
        else:
            result = st.session_state.result
            st.caption("Analyse par Ollama" if result["mode"] == "llm" else "Extraits documentaires sans analyse LLM")
            for warning in result["warnings"]:
                st.info(warning)
            if not result["recommendations"]:
                st.warning("Aucune piste avec preuve suffisante n'a été conservée. Consultez les passages et les informations manquantes.")
            for rank, item in enumerate(result["recommendations"], 1):
                with st.container(border=True):
                    st.subheader(f"{rank}. {item['program_name']}")
                    st.caption(item["status"])
                    st.write(item["reason"])
                    for caution in item["cautions"]:
                        st.write("• " + caution)
                    for citation in item["citations"]:
                        st.caption(f"{citation['source']} — page PDF {citation['page']}")
                        if citation.get("source_text_method") == "table_row_reconstructed_visual_verified":
                            st.caption("Transcription d'une ligne de tableau vérifiée visuellement.")
                        st.text(citation["quote"])
            if result["missing_information"]:
                st.subheader("Informations à préciser pour ce User")
                for missing in result["missing_information"]:
                    st.write("• " + missing)
            with st.expander("Requête et passages retrouvés"):
                st.text(st.session_state.result_query)
                for hit in st.session_state.hits:
                    st.markdown(f"**{hit['program_name']} · page {hit['page']}**")
                    st.text(hit["text"])
                    for caveat in hit.get("caveats", []):
                        st.caption(caveat)
            bundle = {"user": selected_user, "source_workbook": source_name, "generated_at": st.session_state.result_time, "answers": answers, "result": result}
            st.download_button("Exporter les résultats de ce User", json_text(bundle), "recommandations_user.json", "application/json")
        st.download_button("Exporter le profil lu", json_text({"User": selected_user, "answers": answers}), "profil_user.json", "application/json")
    if run_from_user:
        with choose:
            if "result" in st.session_state:
                st.success(f"Analyse terminée pour {selected_user}. Ouvrez l'onglet « 2 · Programmes proposés ».")
                st.write([item["program_name"] for item in st.session_state.result["recommendations"]])
            else:
                st.warning("Consultez le message dans l'onglet « 2 · Programmes proposés ».")

    with library:
        st.subheader("Guide documentaire et catalogue")
        st.caption("Les pages indiquées sont les pages du fichier PDF, à partir de 1.")
        pdf_path = DATA / "raw" / manifest["pdf_name"]
        st.download_button("Télécharger le guide source", pdf_path.read_bytes(), manifest["pdf_name"], "application/pdf")
        category = st.selectbox("Catégorie", ["Toutes"] + sorted({p["category"] for p in catalog}))
        keyword = st.text_input("Nom du programme à rechercher")
        selected = [p for p in catalog if (category == "Toutes" or p["category"] == category) and keyword.casefold() in p["name"].casefold()]
        st.dataframe([{"Programme": p["name"], "Catégorie": p["category"], "Pages PDF": ", ".join(map(str, p["pages"]))} for p in selected], hide_index=True, width="stretch")
        for program in selected[:12]:
            with st.expander(program["name"]):
                st.text(program["source_text"])
                for caveat in program.get("caveats", []):
                    st.caption(caveat)
    with method:
        st.subheader("Comment le User devient une recommandation")
        st.markdown("1. Lecture du classeur : une ligne par User, colonnes Q1 à Q134.\n2. Correspondance avec les libellés du questionnaire Word et validation des réponses.\n3. Construction automatique du profil et d'une requête sur son besoin.\n4. Recherche dans le guide PDF par BM25, avec embeddings locaux en option.\n5. Analyse du profil par Ollama et contrôle des citations avant affichage.")
        st.write("L'Excel fournit les réponses des Users. Le PDF fournit les programmes et leurs conditions. Le Word sert à comprendre les colonnes Q1…Q134. Aucun remplissage manuel n'est nécessaire.")
        st.write("Les profils synthétiques n'ont pas de programme de référence. Les tests de recherche sont illustratifs ; ils ne mesurent pas la précision des recommandations sur les 10 000 Users.")
        st.write("Les citations sont vérifiées textuellement. L'interprétation du LLM et l'éligibilité restent à vérifier. Le guide date de décembre 2025 ; certaines offres sont datées et les tableaux transcrits sont signalés.")
        st.code('python -m unittest discover -s tests -v\npython cli.py recommend --user "User 1" --semantic\npython cli.py evaluate', language="bash")
