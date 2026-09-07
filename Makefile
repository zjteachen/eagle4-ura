.PHONY: lint sim synth clean

lint:
	./scripts/build.sh lint $(TOP)

sim:
	./scripts/build.sh sim $(MODE)

synth:
	./scripts/build.sh synth $(if $(PART),$(PART)) $(if $(TOP),$(TOP)) $(if $(PERIOD),$(PERIOD))

clean:
	./scripts/build.sh clean
