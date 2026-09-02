# BookSearch RAG API

En FastAPI-backend som låter dig ställa frågor om PDF-böcker i `data/pdfs`.
PDF:er hanteras vid uppstart, delas upp i textstycken och lagras som embeddings i PostgreSQL med pgvector.

## RAG-flöde

```text
Läsarfråga
  → Query rewrite
  → flera pgvector-sökningar (semantisk sökning)
  → PostgreSQL fulltextsökning (lexikal sökning)
  → RRF kombinerar båda söktypernas kandidater
  → LLM-reranker väljer bästa styckena
  → RAG-svar med bok- och sidkällor
  → AI-as-judge (valfritt)
```

LangGraph orkestrerar de tre sista huvudstegen: `retrieve` → `generate_answer` →
`judge` (endast när `evaluate: true`). Query rewrite, hybrid search, RRF och
reranking ligger kvar inuti retrieve-noden. Det gör flödet tydligt och ger en bra
grund om du senare vill lägga till exempelvis retry- eller granskningssteg.

### 1. PDF-ingestering

När API:t startar läser det PDF:er i `data/pdfs` med LangChain. Varje sida delas upp i överlappande chunks. Chunks sparas med boknamn, sida och en OpenAI-embedding i pgvector.

En fil som inte har ändrats hoppas över vid nästa uppstart. En ändrad PDF ersätter sina tidigare chunks.

### 2. Query rewrite

Originalfrågan behålls alltid. En liten LLM skapar dessutom upp till två sökvarianter för att hitta relevanta stycken när frågan är formulerad vagt eller med andra ord än i boken.

Om rewrite-anropet misslyckas används enbart originalfrågan.

### 3. pgvector och lexikal sökning

Varje sökvariant används i pgvector för att hitta text med liknande betydelse. Originalfrågan används dessutom i PostgreSQLs fulltextsökning för att hitta exakta namn, datum, citat och andra nyckelord.

Den lexikala sökningen finns i `src/app/lexical_search.py` och har egna tabeller och GIN-index i PostgreSQL. Den är alltså inte beroende av LangChains interna pgvector-tabeller.

### 4. RRF

Reciprocal Rank Fusion (RRF) kombinerar rankingarna från alla semantiska pgvector-sökningar och den lexikala fulltextsökningen. Stycken som rankar högt i flera listor får högre poäng.

### 5. LLM-reranker

Rerankern får de bästa RRF-kandidaterna och originalfrågan. Den väljer de stycken som tydligast stöder ett svar innan de skickas till svarmodellen. Om rerankern misslyckas används RRF-resultatet i stället.

### 6. RAG-svar

Svarmodellen får bara de valda bokutdragen och instruktionen att säga när underlaget inte räcker. Den ska hänvisa till bokfil och sida när den använder ett utdrag.

### 7. AI-as-judge

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
src/app/workflow.py LangGraph-noder och flöde
src/app/query_rewriter.py  Sökfrågevarianter
src/app/lexical_search.py  PostgreSQL fulltextsökning
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

LEXICAL_SEARCH_ENABLED=true
LEXICAL_SEARCH_CONFIG=simple

JUDGE_MODEL=gpt-4.1-mini
```

Sätt `QUERY_REWRITE_ENABLED=false`, `LEXICAL_SEARCH_ENABLED=false` eller `RERANK_ENABLED=false` om du vill jämföra resultat, minska latens eller sänka kostnaden per fråga.

`LEXICAL_SEARCH_CONFIG=simple` passar en blandad svensk/engelsk boksamling. För en enbart svensk samling kan du använda `swedish`; för en enbart engelsk samling `english`. Nästa uppstart bygger då om det lexikala indexet för befintliga PDF:er.

## Begränsningar och nästa steg

- Skannade eller bildbaserade PDF:er behöver OCR för att ge bra text.
- Hybrid search använder PostgreSQL fulltextsökning, inte BM25. Ett externt söksystem behövs först om du senare behöver mer avancerad BM25-, synonym- eller skaleringsfunktionalitet.
- Byte till `text-embedding-3-large` kräver att alla PDF:er embedas om.
- Lägg till autentisering och användarägarskap innan appen exponeras publikt.
