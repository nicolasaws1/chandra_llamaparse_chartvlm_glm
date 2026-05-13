# SB100 Comparação de Métodos de Detecção e Classificação de Conteúdo em PDFs de Agronomia

Notebooks da Sprint 29/abr–13/mai 2026 do projeto **SB100 Agrônomo Virtual** (Squad 02 Ingestão e Vetorização).

Comparação sistemática de cinco métodos de detecção (Chandra OCR 2, PyMuPDF get_images, PyMuPDF get_drawings, GLM-OCR, LlamaParse), dois classificadores de gráficos (ChartVLM-large, LlamaParse) e três métodos com saída textual estruturada (Chandra, GLM-OCR, LlamaParse), aplicados a um corpus de 26 PDFs científicos curados.

## Estrutura dos notebooks

| Notebook | Objetivo | Hardware necessário |
|---|---|---|
| **N1** — Comparação 3 métodos com avaliação manual | Detecção de gráficos com Chandra, PyMuPDF get_images, PyMuPDF get_drawings em 26 PDFs + widget de avaliação manual | Colab T4 |
| **N2** — Comparação 4 métodos (top-10 PDFs difíceis) | Adiciona GLM-OCR (configurado com 3 prompts: Document Parsing, Table Recognition, Text Recognition) | Colab L4 (GLM-OCR ~3 GB VRAM) |
| **N3** — Classificação com ChartVLM-large | Classifica os 170 crops produzidos pelos detectores via ChartVLM (Hugging Face) | Colab L4/A100 (ChartVLM-large ~13B params) |
| **N4** — LlamaParse: detector + classificador + comparação textual | Usa LlamaParse como (A) detector completo, (B) classificador de crops, (C) comparação de qualidade textual com Chandra | Sem GPU (API cloud); requer Colab Secret `LLAMA_CLOUD_API_KEY` |
| **N5** — Comparação de qualidade em 4 dimensões | Consolida outputs dos notebooks anteriores em heatmaps de concordância pareada (Jaccard) entre métodos | Colab T4 |

## Ordem de execução

1. **N1** — gera GT manual + crops do Chandra/PyMuPDF.
2. **N2** — adiciona GLM-OCR (depende dos crops do N1).
3. **N3** — classifica crops com ChartVLM (depende de N1).
4. **N4** — roda LlamaParse independentemente.
5. **N5** — análise final (depende de N1, N2, N3, N4 já terem rodado).

## Estrutura de pastas esperada no Google Drive

```
/MyDrive/Chandra2/
├── PDF/                              # PDFs de entrada (26)
└── output/
    ├── avaliacao_manual.csv          # GT manual (preenchido em N1)
    ├── chandra/<pdf_stem>/           # outputs do Chandra
    ├── pymupdf_raster/<pdf_stem>/    # crops raster do PyMuPDF
    ├── pymupdf_vector/<pdf_stem>/    # crops vetoriais (DBSCAN)
    ├── glm_ocr/<pdf_stem>/output.md  # markdowns do GLM-OCR
    ├── llamaparse/<pdf_stem>/        # markdowns do LlamaParse
    ├── chartvlm_results/             # classificações do ChartVLM
    └── text_comparison/              # análise final do N5
```

## Observações importantes

### dots.ocr 1.5 não está incluído

O dots.ocr 1.5 foi originalmente planejado como detector adicional. Sua execução foi inviabilizada por incompatibilidades técnicas persistentes em Colab:

- Dependência de `flash_attn` (compilação inviável em runtime).
- Parâmetro `mm_token_type_ids` retornado pelo processor mas não aceito pelo `generate` em `transformers ≥ 4.45`.
- Bug em `prepare_inputs_for_generation`: `cache_position[0]` falha quando `cache_position` é `None`.

Mesmo com workarounds aplicados (stub de `flash_attn`, filtro de kwargs, patch no `modeling_dots_ocr.py`), o modelo não retornou inferências válidas. Permanece como prioridade técnica para sprint futura.

### Versões de transformers conflitantes

- **N1, N2, N4, N5** usam `transformers` atualizado.
- **N3 (ChartVLM)** exige `transformers==4.31.0`.

Recomenda-se reset do runtime do Colab entre N3 e os demais notebooks.

### LlamaParse usa async

O N4 usa `aload_data` (async) em vez de `load_data` (sync) para evitar erro `Event loop is closed` em chamadas sequenciais dentro do Colab. A chave da API deve estar em Colab Secrets como `LLAMA_CLOUD_API_KEY` — **nunca hardcoded** no notebook.

### GLM-OCR usa 3 prompts

Após investigação inicial, descobriu-se que o GLM-OCR (0,9B parâmetros) com apenas o prompt `Text Recognition:` descartava completamente figuras e tabelas estruturadas. A versão atual roda **3 prompts em sequência** (`Document Parsing:`, `Table Recognition:`, `Text Recognition:`) e combina os outputs. Tempo de inferência ~3x maior mas resultado comparável aos demais métodos no quesito multimodal.

## Resultados da sprint (em poucas linhas)

- **Tarefa 1 (Classificação de gráficos, critério ≥ 90%):** atendido pelo **LlamaParse** com **91,7% lenient** sobre crops do Chandra. ChartVLM-large ficou em 19,4% strict / 50,0% lenient — inadequado como classificador único.
- **Tarefa 2 (Tabelas):** LlamaParse com MAE 6,3 e viés +2,3 (melhor calibrado); Chandra MAE 5,4 (mais super-detecção); GLM-OCR MAE 11,9 (super-detecção catastrófica).
- **Tarefa 3 (Texto):** concordância Jaccard 86–90% entre os 3 métodos com saída textual.
- **Tarefa 4 (Separação por categoria):** 4 métodos × 4 dimensões comparadas; ver Tab. 3.8 do relatório.

## Autor

**Nicolas Alves Witzel da Silva**
Bolsista de Iniciação Científica FAPESP — Squad 02, Projeto SB100
São Paulo, Brasil, 2026
