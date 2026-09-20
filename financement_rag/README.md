# Financement · Questionnaire, profil, RAG et Ollama

Application Python/Streamlit pour explorer des dispositifs de financement et d'accompagnement au Maroc à partir des documents fournis. Le parcours est **questionnaire → profil structuré → recherche dans le guide → explication par un LLM local → citations contrôlées**.

Le projet contient le code, les trois fichiers sources et leurs données préparées : **134 questions**, un parcours essentiel de **35 questions**, **99 dispositifs** issus d'un PDF de **96 pages**, et le classeur de **10 000 profils synthétiques**. Le guide documentaire est daté de **décembre 2025**. Les réponses sont des pistes à vérifier, jamais une décision d'éligibilité.

Pour une prise en main en darija : [GUIDE_DARIJA.md](GUIDE_DARIJA.md).

## Démarrer sous Windows

Prérequis : **Python 3.10 ou supérieur**, puis [Ollama pour Windows](https://ollama.com/download/windows) pour utiliser le LLM. Ouvrir PowerShell dans le dossier du projet et exécuter :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama list
```

Si les modèles suivants ne figurent pas dans la liste, les télécharger :

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Ouvrir l'adresse locale affichée, généralement `http://127.0.0.1:8501`. L'application Ollama doit être démarrée ; si aucun serveur Ollama ne tourne, `ollama serve` permet de le lancer dans un autre terminal. Ne pas lancer une deuxième instance si le port 11434 est déjà utilisé.

`run_windows.bat` fournit également un lancement guidé : il crée `.venv` si nécessaire, installe les dépendances, puis lance Streamlit. Il ne télécharge pas les modèles Ollama. Les téléchargements initiaux nécessitent Internet ; ensuite le fonctionnement peut être local, avec les dépendances et les modèles déjà présents. Aucune clé API n'est nécessaire.

Les modèles par défaut de l'interface et du CLI sont `qwen2.5:7b-instruct-q4_K_M` et `nomic-embed-text:latest`. Ils peuvent être remplacés dans les réglages. La vitesse dépend de la mémoire, du processeur et du GPU. Nomic est fourni comme option disponible localement : sa pertinence pour ce corpus français doit être évaluée, sans supposer une qualité multilingue particulière. Changer de modèle d'embeddings impose de reconstruire l'index.

## Utiliser l'interface

1. Dans **Réglages Ollama**, cliquer sur **Vérifier Ollama** et contrôler les noms des modèles.
2. Dans **Questionnaire**, choisir **Essentiel** ou **Complet · 134 questions**. Les réponses vides restent inconnues ; les montants sont exprimés en DH.
3. Pour une démonstration, charger un exemple ou saisir `User 1`, puis cliquer sur **Charger ce User depuis Excel**. Le classeur fourni contient les identifiants `User 1` à `User 10000`.
4. Modifier les réponses puis cliquer sur **Enregistrer le profil**. Un fichier JSON de réponses ou un classeur Excel de même structure peut aussi être importé.
5. Garder **RAG + Ollama local**, le mode par défaut. La recherche BM25 fonctionne directement avec le corpus préparé.
6. Pour ajouter la recherche vectorielle, cliquer une fois sur **Construire l'index sémantique**, puis cocher **Recherche hybride avec embeddings**. L'index est réutilisable tant que le corpus et le modèle restent identiques.
7. Dans **Résultats**, cliquer sur **Rechercher les programmes**. Lire les citations, pages PDF, précautions et informations manquantes. Les boutons d'export enregistrent volontairement le profil ou le résultat en JSON.

**Extraits sans LLM** affiche uniquement des passages retrouvés, sans déduire de compatibilité avec le profil. Décocher aussi la recherche hybride pour utiliser ce mode sans aucun appel Ollama. Il n'existe pas de bascule silencieuse vers ce mode : une erreur du LLM ou de l'index est affichée. L'onglet **Sources** permet de parcourir les dispositifs et de consulter le PDF original.

## Architecture et fichiers

```text
app.py                         Interface Streamlit
cli.py                         Commandes reproductibles
finrag/profile.py              Normalisation, profil, requête, lecture Excel
finrag/ingest.py               Extraction PDF et découpage par programme/page
finrag/retrieval.py            BM25, embeddings, fusion RRF et cache JSON
finrag/ollama_client.py        Client HTTP local Ollama
finrag/generation.py           Prompt, schéma JSON et contrôle des citations
data/raw/                     Questionnaire DOCX, classeur XLSX, guide PDF
data/questionnaire_schema.json Questions, options et parcours essentiel
data/program_catalog.json     Catalogue, pages et passages attribués
data/pdf_pages.json           Texte extrait par page
data/manifest.json            Édition, nombres et empreintes des sources
data/pdf_source_metadata.json Métadonnées et limites d'extraction
data/example_profiles.json    Trois profils de démonstration
data/retrieval_cases.json     Huit cas de recherche illustratifs
data/index/embeddings.json    Cache créé lors de l'indexation
tests/                        Tests du corpus, profil, RAG et interface
```

Les passages sont découpés avec recouvrement, tout en conservant leur programme et leur page. BM25 normalise notamment les accents français. Le mode hybride combine les rangs BM25 et cosinus par **Reciprocal Rank Fusion** ; il limite le résultat à deux passages par programme. Ces scores mesurent le classement documentaire, pas une probabilité d'acceptation.

L'index contient les vecteurs du guide, une version, le modèle et une empreinte du corpus. Sa reconstruction remplace le cache uniquement après une vectorisation complète. Un cache absent, corrompu ou incompatible provoque une erreur explicite. Pour Nomic, le moteur applique les préfixes `search_document:` et `search_query:` conformément à la [fiche officielle du modèle](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5).

Ollama reçoit le profil et les passages comme des données, jamais comme des instructions à exécuter. La réponse doit respecter un schéma JSON. Les programmes sont limités aux extraits retrouvés ; les citations doivent y figurer textuellement après normalisation des espaces et appartenir au même programme. Titres, sources et pages proviennent du corpus, pas du LLM. Une piste invalide est écartée avec un avertissement. Sans passage, aucun appel de génération n'est effectué.

## Commandes

Depuis le dossier du projet, utiliser le Python de `.venv` :

```powershell
# État Ollama et construction de l'index
.\.venv\Scripts\python.exe cli.py check
.\.venv\Scripts\python.exe cli.py index
# Exemple intégré : recherche + LLM, puis variante hybride
.\.venv\Scripts\python.exe cli.py recommend
.\.venv\Scripts\python.exe cli.py recommend --semantic
# Réponses exportées depuis l'interface et export explicite du résultat
.\.venv\Scripts\python.exe cli.py recommend --profile profil.json --out recommandations.json
# Recherche documentaire seule, sans appel Ollama
.\.venv\Scripts\python.exe cli.py recommend --excerpts-only
# Réextraction de la même édition du PDF uniquement
.\.venv\Scripts\python.exe cli.py ingest
# Vérification du logiciel et évaluations illustratives
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe cli.py evaluate --k 5
.\.venv\Scripts\python.exe cli.py evaluate --semantic --k 5
.\.venv\Scripts\python.exe cli.py audit-excel --limit 10000 --out audit.json
```

Le CLI accepte `--host`, `--chat-model`, `--embed-model` et `--timeout` **avant** la sous-commande. Exemple : `python cli.py --timeout 600 recommend --top 3 --passages 8`. Les variables `OLLAMA_HOST`, `OLLAMA_CHAT_MODEL` et `OLLAMA_EMBED_MODEL` définissent aussi les valeurs initiales de l'interface et du CLI. Pour les détails : `python cli.py --help` et `python cli.py recommend --help`.

Un profil importable a la forme `{"answers":{"Q1":"réponse","Q2":"réponse"}}`, avec des valeurs conformes aux questions ; les choix multiples sont des listes. Les réponses peuvent également être fournies directement sous les clés `Q1` à `Q134`. L'audit du classeur vérifie les réponses et relève des incohérences sans appeler le LLM.

## Limites et évaluation

- Le guide de décembre 2025 ne permet pas de confirmer les programmes actuellement ouverts, leurs conditions actuelles ou l'acceptation d'un dossier.
- **21 lignes de tableaux ont été reconstruites**, avec une mention dans le catalogue. Le contrôle textuel des citations porte alors sur cette transcription ; consulter la page PDF pour vérifier l'original.
- Une citation exacte prouve sa provenance textuelle. Elle ne valide pas automatiquement le sens de toute l'explication du LLM ni l'absence d'hallucination.
- Le projet ne contient **pas de moteur complet de règles d'éligibilité**, de fine-tuning ou d'entraînement. Les règles du profil servent à signaler des incohérences de saisie.
- Les 10 000 profils sont synthétiques et n'ont pas de recommandations de référence : ils ne permettent pas de calculer une précision de recommandation.
- Les huit cas de `evaluate` comportent notamment les noms exacts de programmes. Recall@k et MRR@k décrivent ces exemples, pas une performance scientifique sur des profils réels. Une étude exige un jeu de profils et de programmes pertinents annoté indépendamment par des personnes compétentes.

Les tests peuvent être relancés avec la commande ci-dessus. Les contrôles de développement ont notamment vérifié le client avec un petit échange JSON Qwen2.5 et des embeddings Nomic locaux ; cela ne garantit ni un temps de réponse ni la qualité des recommandations sur tous les profils.

## Remplacer le guide ou faire évoluer le questionnaire

`cli.py ingest --pdf ...` accepte uniquement le PDF correspondant à l'empreinte du manifeste. Un PDF différent requiert une mise à jour explicite : extraire les nouvelles pages, reconstruire le catalogue et l'association programme/page, vérifier les tableaux et transcriptions, puis actualiser le manifeste et les métadonnées ensemble. Ne pas recopier les anciens numéros de page ni les anciens extraits dans une nouvelle édition. Adapter les cas de vérification, relancer les tests, reconstruire les embeddings et redémarrer Streamlit pour vider son cache de corpus.

Pour modifier le questionnaire, mettre à jour `questionnaire_schema.json`, les correspondances du profil et de la requête dans `profile.py`, les profils exemples et les tests. Préserver ou migrer explicitement les identifiants `Q…` utilisés dans les exports.

## Données et dépannage

L'application garde les réponses en mémoire de session et ne crée pas de fichier de profil automatiquement. Les exports et `--out` sont des enregistrements volontaires. Le cache disque concerne le guide documentaire. Le client n'utilise ni clé API, ni proxy système, ni redirection HTTP ; il accepte uniquement une adresse de boucle locale et refuse les noms de modèles explicitement marqués cloud. Utiliser des modèles réellement téléchargés et vérifier séparément les réglages et journaux des logiciels tiers.

| Problème | Action |
|---|---|
| Ollama inaccessible | Démarrer Ollama, puis vérifier `http://127.0.0.1:11434` et `ollama list`. |
| Modèle introuvable | Copier le nom exact de `ollama list` dans les réglages, ou télécharger le modèle. |
| Index absent ou incompatible | Reconstruire l'index avec le modèle d'embeddings sélectionné, ou décocher le mode hybride. |
| Délai dépassé | Vérifier les ressources, choisir un modèle local plus léger ou augmenter le délai du CLI. |
| JSON ou citations rejetés | Relancer la génération ou consulter les extraits ; ne pas désactiver la validation. |

Le client suit les API officielles [génération d'embeddings](https://docs.ollama.com/api/embed), [conversation](https://docs.ollama.com/api/chat) et [sorties structurées](https://docs.ollama.com/capabilities/structured-outputs). Il désactive la troncature silencieuse des embeddings et demande une réponse JSON sans streaming.
