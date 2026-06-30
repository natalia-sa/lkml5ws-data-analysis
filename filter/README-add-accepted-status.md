# `add_accepted_status.py` — coluna `accepted` via Patchwork

Script que lê os CSVs `*-duplicated` e gera versões `*-status` com uma coluna nova
(`accepted`) indicando se cada patch foi **aceito** no seu subsistema, consultando o
Patchwork.

---

## Como usar

```bash
cd /home/natalia/Documentos/dataset
.venv/bin/python filter/add_accepted_status.py
```

Precisa de acesso à internet (bate em APIs públicas). Sem dependências externas — só a
biblioteca padrão do Python (`urllib`).

### Entradas e saídas

| Entrada | Saída |
|---|---|
| `filter/iio-duplicated.csv` | `filter/iio-duplicated-status.csv` |
| `filter/amd-duplicated.csv` | `filter/amd-duplicated-status.csv` |

Os arquivos originais **não** são alterados. A saída é idêntica à entrada, com **todas as
colunas originais na mesma ordem** + uma coluna nova `accepted` no final.

### Reexecução / cache

As respostas do Patchwork são gravadas em `filter/.patchwork_cache.json`, com checkpoint a
cada consulta. Rodar de novo:

- **não rebate** a API para msgids já consultados (fica instantâneo);
- é seguro interromper no meio — o progresso já consultado fica salvo.

Para forçar reconsulta do zero, apague `filter/.patchwork_cache.json`.

### Tempo de execução

~1 chamada/segundo (throttle educado). Na 1ª execução: iio ≈ 64 chamadas, amd ≈ 232
chamadas (~5 min no total). Execuções seguintes saem do cache.

---

## O que significa a coluna `accepted`

A coluna é preenchida **apenas na linha do último patch de cada patchset** (ver "Como
funciona"). Nas demais linhas ela fica **vazia**.

| Valor | Significado |
|---|---|
| `True` | O Patchwork marca o patch como **`accepted`** (entrou na árvore do subsistema). |
| `False` | O patch **foi encontrado**, mas está em outro estado: `new`, `superseded`, `rejected`, `changes-requested`, `handled-elsewhere`, `not-applicable`, etc. Ou seja: **não consta como aceito**. |
| *(vazio)* | Duas situações: (a) a linha **não é** o último patch de um patchset; ou (b) é, mas o **msgid não foi encontrado** no Patchwork (msgid anonimizado no dataset, ou patch não indexado). |

> Importante: `False` **não** quer dizer necessariamente "rejeitado". Quer dizer "não está
> marcado como aceito". Inclui patches ainda pendentes (`new`), substituídos por uma versão
> nova (`superseded`), e os de fato rejeitados (`rejected`). Da mesma forma, `vazio` quando o
> msgid não é achado significa "não sei", e não "não foi aceito".

### E as várias versões (v1, v2, v3)?

Não há passo de ligar versões. Cada versão é um patchset separado e é consultada por conta
própria. O **próprio estado do Patchwork** já resolve isso: versões antigas voltam como
`superseded` (→ `False`) e a versão final volta `accepted` (→ `True`) quando aplicada.

---

## Em quais APIs ele bate

Os dois subsistemas vivem em **instâncias diferentes** do Patchwork, com métodos diferentes.

### iio → `patchwork.kernel.org` (API REST moderna)

1. `GET https://patchwork.kernel.org/api/patches/?msgid=<message_id>`
2. O mesmo msgid pode existir em vários projetos; o script filtra o item cujo
   `project.list_id == "linux-iio.vger.kernel.org"` (projeto "Linux IIO").
3. Lê o campo `state` (já vem como **string**: `accepted`, `new`, `superseded`, ...).

`accepted = (state == "accepted")`.

### amd → `patchwork.freedesktop.org` (fork antigo, API `1.0`)

Nessa instância o filtro `?msgid=` é ignorado, então o caminho é em dois passos:

1. `GET https://patchwork.freedesktop.org/patch/msgid/<message_id>/`
   → responde `302` redirecionando para `/patch/<id>/`. O script extrai o `<id>`.
2. `GET https://patchwork.freedesktop.org/api/1.0/patches/<id>/`
   → lê o campo `state`, que aqui é um **inteiro**. Mapa usado:

   | id | estado |
   |----|--------|
   | 1 | new |
   | 3 | accepted |
   | 4 | rejected |
   | 9 | superseded |

`accepted = (state == 3)`.

(Projeto amd-gfx = id 22, lista `amd-gfx@lists.freedesktop.org`.)

---

## Como funciona (passo a passo)

1. **Lê o CSV** preservando todas as colunas e a ordem das linhas.
2. **Identifica submissões de patch**: linha com `has_patch_tag == True` **e** cujo subject
   **não** começa com `Re:`. (A flag `has_patch_tag` sozinha não serve — ela também é `True`
   nas respostas `Re: [PATCH...]`.)
3. **Agrupa em patchsets**: submissões com o mesmo `_thread_id` e mesma `patch_version`.
4. **Acha o último patch** de cada patchset: maior numerador de `patchset_sequence_number`
   (`"2/2"` → 2); patch único sem `n/m` é o próprio último; a cover letter `0/m` nunca é o
   último.
5. **Consulta o Patchwork** com o `message_id` desse último patch (iio ou amd conforme o
   arquivo), usando cache.
6. **Preenche `accepted`** (`True`/`False`/vazio) só naquela linha.
7. **Escreve** o CSV novo `*-status.csv` e imprime um resumo (quantos True/False/vazio).

---

## Limitações conhecidas

- `accepted` reflete a **marcação manual do maintainer** no Patchwork. Um patch que entrou na
  árvore mas não foi marcado como `accepted` aparece como `False`.
- Para séries multi-patch, o valor reflete o estado do **último patch** da série, não da série
  inteira.
- Msgids anonimizados no dataset (ex.: `uuid@gmail.com`) não existem no Patchwork → `vazio`.
- Depende das APIs públicas estarem no ar e dos limites de taxa do `patchwork.kernel.org`
  (mitigado pelo throttle + retry com backoff).

---

## Ajustes rápidos (no topo do script)

- `THROTTLE_SECONDS` — intervalo entre chamadas (padrão `1.0`).
- `CONFIG` — pares (entrada, saída, subsistema).
- `FREEDESKTOP_STATES` — mapa id→estado do freedesktop, caso apareça um id novo.
- `CACHE_PATH` — local do cache.
