.PHONY: test check

test:
	pytest --disable-warnings

check:
	pyflakes **.py
	pycodestyle **.py

upload:
	python setup.py sdist
	twine upload dist/*
