.PHONY: test check

test:
	pytest --disable-warnings

check:
	pyflakes **/*.py
	pycodestyle --ignore=E501 **/*.py

upload:
	python setup.py sdist
	twine upload dist/*
