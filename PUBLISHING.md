# Publishing qutip-bundling and getting it listed by QuTiP

Everything below is done from your own machine with your own accounts; none of
it can be done from inside this environment because it needs your credentials.

--------------------------------------------------------------------------
## A. Put the code on GitHub

1. Create a new public repo, e.g. `qutip-bundling`, under your account.
2. In `pyproject.toml`, replace `Sayak183` in the two GitHub URLs with
   your actual GitHub username (or org).
3. Push the package:

   ```bash
   cd qutip-bundling
   git init
   git add .
   git commit -m "Initial release: stochastic bundling of Lindblad operators"
   git branch -M main
   git remote add origin https://github.com/Sayak183/qutip-bundling.git
   git push -u origin main
   ```

4. (Recommended) Turn on GitHub Actions CI so the tests run on every push.
   A minimal workflow at `.github/workflows/test.yml`:

   ```yaml
   name: tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with: { python-version: "3.11" }
         - run: pip install -e ".[test]"
         - run: pytest -q
   ```

--------------------------------------------------------------------------
## B. Publish to PyPI

You need a free account at https://pypi.org (and ideally test first on
https://test.pypi.org).

1. Install the build/upload tools:

   ```bash
   pip install build twine
   ```

2. Build the distributions (creates `dist/*.whl` and `dist/*.tar.gz`):

   ```bash
   python -m build
   ```

3. (Optional but smart) upload to TestPyPI first and check it installs:

   ```bash
   twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ qutip-bundling
   ```

4. Upload to the real PyPI:

   ```bash
   twine upload dist/*
   ```

   Twine will ask for an API token. Create one at
   https://pypi.org/manage/account/token/ and paste it (username is
   `__token__`).

5. Confirm:

   ```bash
   pip install qutip-bundling
   python -c "import qutip_bundling as qb; print(qb.__version__)"
   ```

Notes
- The project name `qutip-bundling` must be unique on PyPI. If taken, pick
  another (e.g. `qutip-stochastic-bundling`) and update `name` in
  `pyproject.toml`.
- Bump the `version` in `pyproject.toml` AND in
  `src/qutip_bundling/__init__.py` for every release; PyPI refuses to
  overwrite an existing version.

--------------------------------------------------------------------------
## C. (Optional) Archive a citable version with Zenodo

For a DOI on the software itself (nice alongside the paper):
1. Link your GitHub repo to https://zenodo.org (GitHub settings -> Zenodo).
2. Cut a GitHub Release (tag e.g. `v0.5.0`). Zenodo mints a DOI automatically.
3. Add that DOI to the README and CITATION.cff.

--------------------------------------------------------------------------
## D. Get it listed by QuTiP

QuTiP maintains a category of "associated"/family packages — third-party
packages built on QuTiP that are linked from the project. You do NOT need to
merge anything into QuTiP core (that would be a much higher bar and is not the
right home for a research method).

The current, lightweight path:

1. Make sure the package is public on GitHub, pip-installable from PyPI, has a
   README, a license (BSD-3 — matches QuTiP's), tests, and at least one
   example/tutorial. (You have all of these.)

2. Open a thread with the QuTiP maintainers. Best venues, in order:
   - The QuTiP GitHub Discussions / community board:
     https://github.com/qutip  (look for the `qutip` org discussions or the
     admin/website repo).
   - The QuTiP Google Group / mailing list (linked from https://qutip.org).
   - Their community chat if one is linked from qutip.org.

3. Ask specifically to be listed as an associated/ecosystem package on
   qutip.org. A short, concrete message works best (draft below).

4. If they maintain a tutorials repository (`qutip/qutip-tutorials`), consider
   contributing your notebook there as well — tutorials are browsed heavily and
   are an easy, welcome contribution.

--------------------------------------------------------------------------
## E. Draft message to the QuTiP maintainers

> Subject: Request to list an associated package: qutip-bundling
>
> Hi QuTiP team,
>
> I've released a small open-source package built on QuTiP that implements the
> stochastically bundled dissipator method (Adhikari & Baer, J. Chem. Theory
> Comput. 2025, 21, 4142, https://doi.org/10.1021/acs.jctc.5c00145). It reduces
> the cost of Lindblad master-equation propagation when the number of collapse
> operators is large, by replacing them with a small set of randomly bundled
> operators whose dissipator equals the full one in expectation.
>
> It depends on QuTiP, is BSD-3 licensed, pip-installable
> (`pip install qutip-bundling`), and ships with tests and an executed tutorial
> notebook. Repo: https://github.com/Sayak183/qutip-bundling
>
> Would you consider listing it as an associated/ecosystem package on
> qutip.org? I'd also be happy to contribute the tutorial to qutip-tutorials if
> that's useful. Thanks for QuTiP — it's been the natural environment to build
> this in.
>
> Best,
> Sayak Adhikari

--------------------------------------------------------------------------
## F. Quick pre-flight checklist

- [ ] Replace Sayak183 in pyproject.toml URLs
- [ ] Confirm author names/emails in pyproject.toml and CITATION.cff
- [ ] `pytest -q` passes locally (36 tests)
- [ ] `python -m build` succeeds
- [ ] README renders correctly on GitHub
- [ ] Tag a release (v0.5.0) once pushed
- [ ] (optional) TestPyPI dry run before real PyPI
