CORE_JS = $(wildcard js/*.js)

AWK=awk

# TODO: for production should use
# JSMIN=uglifyjs
# $(JSMIN) $^ -m > static/core-min.js

all: js

js: static/core-min.js

static/core-min.js: $(CORE_JS)
	$(AWK) 1 $^ > static/core-min.js

clean:
	rm -f static/core-min.js
