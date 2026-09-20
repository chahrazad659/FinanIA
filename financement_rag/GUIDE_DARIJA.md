# كيفاش تشغّلي المشروع

هاد المشروع فيه **Python + Streamlit + Ollama محلي**. كتعمرّي questionnaire، كيتبنى profil، كيتقلب فالدليل بطريقة RAG، ومن بعد LLM كيقترح برامج مع المقاطع والصفحات اللي اعتمد عليها.

## التشغيل أول مرة

خاص Python 3.10 ولا أكثر و[Ollama](https://ollama.com/download/windows). حلّي PowerShell داخل dossier ديال المشروع ونفّذي:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

إلا الموديلات موجودين فـ`ollama list` ما محتاجاش تعاودي تحميلهم. التحميل الأول خاصو الإنترنت؛ من بعد تقدري تخدمي محلياً بالموديلات والمكتبات اللي تحملو. ما كاينة حتى API key. خلي Ollama خدام وحلّي الرابط اللي كيبان فالترمينال، غالباً `http://127.0.0.1:8501`.

## تجربة سريعة

1. كليكي على **Vérifier Ollama** من الجنب.
2. فـ**Questionnaire**، حلّي **Charger un exemple ou importer un profil** واختاري مثال، أو دخلي `User 1` وكليكي **Charger ce User depuis Excel**.
3. تقدري تعمري **Essentiel** فيه 35 سؤال، أو **Complet** فيه 134. من بعد التعديل، كليكي **Enregistrer le profil**.
4. خلي **RAG + Ollama local** باش LLM يفسّر النتائج. البحث بالكلمات BM25 خدام مباشرة.
5. باش تزيدي البحث بالمعنى، كليكي **Construire l'index sémantique** مرة وحدة ومن بعد فعّلي **Recherche hybride avec embeddings**.
6. مشي لـ**Résultats** وكليكي **Rechercher les programmes**. قراي المصادر، الصفحات، والمعلومات اللي باقي خاصها تتأكد. تقدري تصدّري profil والنتيجة بصيغة JSON.

إلا بغيتي غير تشوفي المقاطع بلا LLM، اختاري **Extraits sans LLM**. باش تخدمي حتى بلا Ollama، حيّدي كذلك **Recherche hybride avec embeddings**. هاد الوضع ما كيحكمش واش البرنامج مناسب للبروفيل؛ كيعرض غير المقاطع اللي لقاها.

## شنو كيدير المشروع وشنو خاصك تعرفي

الدليل فيه 99 dispositif فـ96 صفحة، وراجع لدجنبر 2025. النتائج اقتراحات خاصها التأكد مع المؤسسة؛ ماشي قبول ديال التمويل وماشي إثبات الأهلية. كاينين 21 سطر من الجداول تعاودو تكتبو؛ راجعي الصفحة الأصلية إلا احتجتي تتأكدي من المعطيات.

الكود كيتأكد أن citation موجودة فالمقطع ومن نفس البرنامج. هاد التحقق ما كيضمنش أن التفسير كامل صحيح. ما كاين لا entraînement، لا fine-tuning، لا moteur كامل ديال règles d'éligibilité.

10 000 profils فالـExcel اصطناعيين، وما عندهمش برامج صحيحة معلّمة كمرجع. تقدري تستعمليهم لتجربة الاستيراد، ولكن ما تقدريش تعتبريهم وحدهم قياس دقة التوصيات. حتى 8 أمثلة ديال evaluation توضيحية وفيها أسماء برامج مباشرة.

الأجوبة كتبقى فذاكرة الجلسة، وما كتتسجلش فملف تلقائياً. Export هو اللي كيخليك تسجليها. إلا بدلتي موديل embeddings ولا الدليل، خاص تعاودي index؛ ودليل جديد خاصو كذلك catalogue وmanifest جداد ومراجعين.

التفاصيل التقنية، أوامر CLI، والاختبارات كاينين فـ[README.md](README.md).
