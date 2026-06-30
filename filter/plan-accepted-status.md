# Plano: coluna `accepted` via Patchwork

## Objetivo

Um **script Python novo** (rodado manualmente) que processa `iio-duplicated.csv` e
`amd-duplicated.csv` e gera novos CSVs com **uma coluna nova**:

- `accepted` — `True`/`False` se o patch foi aceito no subsistema, preenchida **só na linha do
  último patch de cada patchset**. **Vazia** quando o msgid não é encontrado no Patchwork
  (anonimizado / não indexado) e em todas as outras linhas.

Saída: CSVs novos, sem sobrescrever os atuais.

| Entrada | Saída |
|---|---|
| `filter/iio-duplicated.csv` | `filter/iio-duplicated-status.csv` |
| `filter/amd-duplicated.csv` | `filter/amd-duplicated-status.csv` |

> Sobre versões: não há passo de ligar v1↔v2↔v3. Cada patchset (thread + versão) é tratado
> independentemente. O próprio estado do Patchwork já resolve o caso de múltiplas versões — versões
> antigas voltam como `superseded` (→ `accepted=False`) e a final volta `accepted` (→ `True`).

---

## Fatos dos dados (já verificados)

- Cada linha é um e-mail. `has_patch_tag=True` aparece **também nas respostas** (`Re: [PATCH...]`),
  então **não** identifica a submissão do patch. Submissão real = `has_patch_tag=True` **e** subject
  que **não** começa com `Re:`.
- Patchset/série: `patchset_sequence_number` no formato `n/m` (ex. `2/2`); patch único (`[PATCH]`
  sem `n/m`) é patchset de 1 → ele mesmo é o último.
- Versão: `patch_version` (ex. `2.0`, `3.0`); vazio = v1. Versões diferentes ficam em threads
  diferentes (`_thread_id` distinto) → cada uma é seu próprio patchset.
- Colunas existentes incluem: `message_id`, `subject`, `from`, `date`, `patch_version`,
  `patchset_sequence_number`, `x_mailing_list`, `_thread_id`, etc.

---

## APIs verificadas (resumo)

Os dois subsistemas vivem em **instâncias diferentes** do Patchwork e exigem **métodos diferentes**.

### IIO → `patchwork.kernel.org` (API moderna)

- `GET https://patchwork.kernel.org/api/patches/?msgid=<msgid>` (msgid **sem** `<>`).
- Retorna lista (o mesmo msgid existe em vários projetos). **Filtrar** pelo item com
  `project.list_id == "linux-iio.vger.kernel.org"` (projeto "Linux IIO", id 359).
- `state` já vem como **string**: `accepted`, `new`, `changes-requested`, `rejected`,
  `superseded`, `not-applicable`, ...
- Aceito ⇔ `state == "accepted"`.

### AMD (amd-gfx) → `patchwork.freedesktop.org` (fork antigo, `revision 5`)

- O filtro `?msgid=` é **ignorado** (devolve tudo). `?q=` não existe. Só `?project=` funciona.
- Caminho que funciona: **redirect por msgid**
  `GET https://patchwork.freedesktop.org/patch/msgid/<msgid>/` → `302` para `/patch/<id>/`
  (seguir o redirect e extrair `<id>`).
- Depois `GET https://patchwork.freedesktop.org/api/1.0/patches/<id>/` → `state` vem como
  **inteiro**. Mapa derivado por teste: **1=New, 3=Accepted, 4=Rejected, 9=Superseded**.
- Aceito ⇔ `state == 3`.
- Projeto amd-gfx = id 22, `amd-gfx@lists.freedesktop.org`.

> Cobertura: a maioria dos msgids de submissão é real e resolvível; uma fração é anonimizada
> (`uuid@gmail.com`) e não resolve → fica `accepted` vazio.

---

## Etapas do script

### 0. Setup
- Deps: `requests` (ou `urllib`), `csv`. Usar `.venv` do projeto (adicionar `requests` se preciso,
  ou usar `urllib`).
- Config no topo: pares (input, output, subsistema) para iio e amd.

### 1. Carregar e marcar submissões de patch
- Ler o CSV preservando **todas** as colunas e a **ordem das linhas** (a saída mantém ordem).
- Marcar `is_submission = has_patch_tag and not subject.lower().startswith("re:")`.

### 2. Agrupar em patchsets e achar o último patch
- Patchset = submissões dentro de um mesmo `_thread_id` com a mesma `patch_version`.
- **Último patch** do patchset = maior numerador de `patchset_sequence_number` (`n/m`);
  patch único (sem `n/m`) é o próprio último. Cover letter `0/m` nunca é o último.
- Guardar, por patchset, o índice da linha do último patch (a linha que receberá `accepted`).

### 3. Consultar Patchwork e preencher `accepted`
- Para cada linha-alvo (último patch do patchset), pegar o `message_id` e consultar conforme o
  subsistema:
  - **iio** → `api/patches/?msgid=`, filtrar projeto `linux-iio`, ler `state` (string).
  - **amd** → redirect `patch/msgid/<id>/` → id → `api/1.0/patches/<id>/`, ler `state` (int).
- Normalizar:
  - encontrado **e** `accepted` → `accepted = True`
  - encontrado **e** outro estado (new/superseded/rejected/changes-requested/...) → `accepted = False`
  - **não encontrado** (404 / lista vazia / msgid anonimizado) → `accepted` **vazio**
- **Cache** local em disco por `(subsistema, msgid)` (ex. `filter/.patchwork_cache.json`) para
  poder reexecutar sem rebater a API.
- **Throttle**: `time.sleep` entre chamadas + retry com backoff em `429`/timeout
  (kernel.org tem rate limit anônimo).

### 4. Escrever os CSVs novos
- Mesmas colunas + `accepted`, mesma ordem das linhas de entrada.
- Gravar em `*-status.csv`. Não tocar nos CSVs originais.
- Logar no final: nº de patchsets, nº de linhas-alvo, quantos accepted=True/False/vazio,
  quantos não encontrados.

---

## Casos de borda e limitações

- **`accepted` não é um booleano perfeito**: "accepted" é marcação manual do maintainer; um patch
  que entrou na árvore mas não foi marcado fica `False`. (Limitação aceita pelo escopo.)
- **Múltiplas versões**: tratadas implicitamente — cada patchset é consultado; versões antigas
  voltam `superseded` (→ `False`) e a final `accepted` (→ `True`). Sem ligação explícita entre elas.
- **Msgids anonimizados** não resolvem → `accepted` vazio (decisão do usuário).
- **Séries multi-patch**: o `accepted` reflete o estado do **último patch** da série (não da série
  inteira).

---

## Estrutura sugerida do código (um arquivo)

```
filter/add_accepted_status.py
  CONFIG = [(input, output, subsystem), ...]
  load_rows(path) -> (rows, fieldnames)
  is_submission(row) -> bool
  parse_seq(s) -> (num, den)                 # "2/2" -> (2,2)
  build_patchsets(rows) -> list[Patchset]    # thread+versão, índice do último patch
  query_iio(msgid) -> state|None             # kernel.org
  query_amd(msgid) -> state|None             # freedesktop (redirect + /api/1.0)
  state_to_accepted(state, subsystem) -> True|False|None
  PatchworkCache (json em disco) + throttle/backoff
  main(): para cada CONFIG -> processa -> escreve *-status.csv
```

## Como rodar / pronto quando

```
.venv/bin/python filter/add_accepted_status.py
```

- Gera `iio-duplicated-status.csv` e `amd-duplicated-status.csv` com a coluna `accepted`.
- `accepted` preenchida só no último patch de cada patchset; vazia se não achado.
- Reexecução usa cache e não rebate a API.
