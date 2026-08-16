.PHONY: lint synth clean

lint:
	./scripts/build.sh lint $(TOP)

synth:
	./scripts/build.sh synth $(if $(PART),$(PART)) $(if $(TOP),$(TOP)) $(if $(PERIOD),$(PERIOD))

clean:
	./scripts/build.sh clean
