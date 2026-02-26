# UV Environment Setup

This project uses `uv` for fast Python package management. The environment has been successfully created and configured.

## Environment Details

- **Python Version**: 3.11.4
- **Virtual Environment**: `.venv/`
- **Package Manager**: uv 0.10.2
- **Total Packages**: 136

## Installed Packages

### Core Dependencies
- langgraph: 1.0.8
- langchain: 1.2.10
- langchain-community: 0.4.1
- langchain-core: 1.2.11
- dashscope: 1.25.12
- chromadb: 1.5.0

### Document Processing
- pypdf: 6.7.0
- openpyxl: 3.1.5
- pandas: 3.0.0

### Utilities
- python-dotenv: 1.2.1
- pydantic: 2.12.5
- numpy: 2.4.2
- scikit-learn: 1.8.0
- tqdm: 4.67.3

### Reranking & Search
- rank-bm25: 0.2.2
- sentence-transformers: 5.2.2
- cohere: 5.20.5

## Activation

### Windows (PowerShell)
```powershell
.venv\Scripts\activate
```

### Windows (Command Prompt)
```cmd
.venv\Scripts\activate.bat
```

## Running the Project

After activation, run the main script:

```bash
python main.py
```

Or in interactive mode:

```bash
python main.py --interactive
```

## Reinstalling Dependencies

If you need to reinstall all dependencies:

```bash
uv pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and configure your Dashscope API key:

```bash
copy .env.example .env
```

Then edit `.env` and add:
```
DASHSCOPE_API_KEY=your_api_key_here
```

## Benefits of Using UV

- **Fast**: Up to 10-100x faster than pip
- **Reliable**: Better dependency resolution
- **Modern**: Written in Rust for performance
- **Compatible**: Works with existing pip workflows

## Troubleshooting

### If activation doesn't work

Try using the full path to Python:

```bash
.venv\Scripts\python.exe main.py
```

### If packages are missing

Reinstall dependencies:

```bash
uv pip install -r requirements.txt --reinstall
```

### Check installed packages

```bash
uv pip list
```

## Next Steps

1. Activate the virtual environment
2. Copy `.env.example` to `.env` and add your API key
3. Run `python main.py --interactive` to start using the RAG system
