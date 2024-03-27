# Static type checking
.PHONY: type-check
type-check:
	mypy .

# Run tests
.PHONY: test
test:
	pytest .

# Format code
.PHONY: format
format:
	black .
	isort .
