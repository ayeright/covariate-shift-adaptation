BUILDDIR      = _build
SPHINXBUILD  ?= sphinx-build
ALLSPHINXOPTS = -d $(BUILDDIR)/doctrees .

all: html

html:
	rm -rf $(BUILDDIR)/html/_images
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html/stable
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html/stable"
