# 1. bump up version
rm -rf dist
python -m build
twine upload --repository pypi dist/*
