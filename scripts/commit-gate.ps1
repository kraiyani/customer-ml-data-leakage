$ErrorActionPreference = 'Stop'
python -m ruff format --check src tests scripts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m ruff check src tests scripts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m ty check --extra-search-path (python -m site --user-site) src scripts
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest
exit $LASTEXITCODE
