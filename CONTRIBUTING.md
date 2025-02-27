# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/jaxtyping.git
cd jaxtyping
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These hooks use ruff to lint and format the code.

Now make your changes. Make sure to include additional tests if necessary.

Next verify the tests all pass:

```bash
pip install -r test/requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pytest ./test
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!
