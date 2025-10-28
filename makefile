
---

### 5) `Makefile`
Save as `Makefile`:

```makefile
.PHONY: install run test format

install:
	@echo "Install project dependencies (dev extras)"
	# if using uv
	uv sync || python -m pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest -q

format:
	pre-commit run --all-files || true
