# BookSearch RAG API

En FastAPI-backend som låter dig ställa frågor om PDF-böcker i `data/pdfs`.
PDF:er hanteras vid uppstart, delas upp i textstycken och lagras som embeddings i PostgreSQL med pgvector.

## RAG-flöde

```text
Läsarfråga
  → Query rewrite
  → flera pgvector-sökningar
  → RRF kombinerar kandidater
  → LLM-reranker väljer bästa styckena
  → RAG-svar med bok- och sidkällor
  → AI-as-judge (valfritt)
```

### 1. PDF-ingestering

När API:t startar läser det PDF:er i `data/pdfs` med LangChain. Varje sida delas upp i överlappande chunks. Chunks sparas med boknamn, sida och en OpenAI-embedding i pgvector.

En fil som inte har ändrats hoppas över vid nästa uppstart. En ändrad PDF ersätter sina tidigare chunks.

### 2. Query rewrite

Originalfrågan behålls alltid. En liten LLM skapar dessutom upp till två sökvarianter för att hitta relevanta stycken när frågan är formulerad vagt eller med andra ord än i boken.

Om rewrite-anropet misslyckas används enbart originalfrågan.

### 3. pgvector och RRF

Varje sökvariant används i pgvector. Resultatlistorna kombineras med Reciprocal Rank Fusion (RRF), som belönar stycken som rankar högt i flera sökningar.

### 4. LLM-reranker

Rerankern får de bästa RRF-kandidaterna och originalfrågan. Den väljer de stycken som tydligast stöder ett svar innan de skickas till svarmodellen. Om rerankern misslyckas används RRF-resultatet i stället.

### 5. RAG-svar

Svarmodellen får bara de valda bokutdragen och instruktionen att säga när underlaget inte räcker. Den ska hänvisa till bokfil och sida när den använder ett utdrag.

### 6. AI-as-judge

Judge-steget är avstängt som standard eftersom det gör ett extra modell-anrop. När det aktiveras bedömer det svaret mot exakt de bokutdrag som användes.

Bedömningen består av:

- groundedness — 50 %
- relevans — 25 %
- källhänvisningar — 15 %
- tydlighet — 10 %

Den returnerar en poäng mellan 0 och 100, plus `pass`, `review` eller `fail`.

## Projektstruktur

```text
data/pdfs/          PDF-böcker att indexera
database/setup.sql  Aktiverar pgvector i den lokala databasen
src/app/main.py     FastAPI-endpoints
src/app/rag.py      Ingestering, pgvector-sökning och RRF
src/app/query_rewriter.py  Sökfrågevarianter
src/app/reranker.py        Reranker efter RRF
src/app/judge.py           AI-as-judge
src/app/config.py   Inställningar från .env
```

## Lokal installation

1. Installera och starta PostgreSQL lokalt med pgvector tillgängligt.
2. Skapa databasen `booksearch` och kör `database/setup.sql` i DBeaver.
3. Kopiera `.env.example` till `.env` och ange din OpenAI API-nyckel samt rätt `DATABASE_URL`.
4. Skapa och aktivera en Python-miljö:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

5. Placera PDF:er i `data/pdfs`.
6. Starta API:t:

   ```powershell
   fastapi dev src/app/main.py
   ```

Öppna sedan `http://127.0.0.1:8000/docs` för FastAPI:s interaktiva API-dokumentation.

## API

### `POST /chat`

```json
{
  "question": "Vad är bokens viktigaste budskap?",
  "book": "min-bok.pdf",
  "top_k": 5,
  "evaluate": true
}
```

- `book` begränsar sökningen till en specifik PDF och är valfri.
- `top_k` är antalet slutliga, rerankade stycken i RAG-kontexten.
- `evaluate` aktiverar AI-as-judge för just det svaret.

Svaret innehåller `answer`, de använda `sources` och, när `evaluate` är `true`, ett `evaluation`-objekt.

### Övriga endpoints

- `GET /health` — kontrollera att API:t körs.
- `GET /books` — lista indexerade PDF:er.
- `POST /evaluate` — utvärdera ett redan genererat svar genom att skicka `question`, `answer` och dess `sources`.

## Inställningar

De viktigaste inställningarna i `.env`:

```env
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small

QUERY_REWRITE_ENABLED=true
QUERY_REWRITE_MODEL=gpt-4.1-mini

RERANK_ENABLED=true
RERANK_MODEL=gpt-4.1-mini
RERANK_CANDIDATE_COUNT=20

JUDGE_MODEL=gpt-4.1-mini
```

Sätt `QUERY_REWRITE_ENABLED=false` eller `RERANK_ENABLED=false` om du vill jämföra resultat, minska latens eller sänka kostnaden per fråga.

## Begränsningar och nästa steg

- Skannade eller bildbaserade PDF:er behöver OCR för att ge bra text.
- Nuvarande sökning är semantisk pgvector-sökning; hybrid search med PostgreSQL fulltextsökning är ännu inte implementerad.
- Byte till `text-embedding-3-large` kräver att alla PDF:er embedas om.
- Lägg till autentisering och användarägarskap innan appen exponeras publikt.
