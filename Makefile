# Common commands. Run `make <target>` from the project root.
.PHONY: install run test eval eval-quick eval-retrieval eval-baseline readme-results docker-build docker-run

install: ## install app + dev dependencies into the active venv
	pip install -r requirements-dev.txt

run: ## start the API with auto-reload (http://127.0.0.1:8000/docs)
	uvicorn app.main:app --reload

test: ## fast unit tests, no API key needed
	python -m pytest -q

eval: ## full evaluation + report vs baseline; exits 1 if below a threshold (stop the API first)
	python -m evals.run_evals

eval-quick: ## full evaluation on the first 3 questions
	python -m evals.run_evals --limit 3

eval-retrieval: ## retrieval-only evaluation (hit@1, hit@k), almost free
	python -m evals.run_evals --retrieval-only

eval-baseline: ## full evaluation, then save it as the new baseline (commit evals/baseline.json)
	python -m evals.run_evals --set-baseline

readme-results: ## write the baseline numbers into the README results table
	python -m evals.readme_results

docker-build: ## build the container image
	docker build -t rag-doc-assistant .

docker-run: ## run the container on port 8000 with your .env
	docker run --env-file .env -p 8000:8000 rag-doc-assistant
