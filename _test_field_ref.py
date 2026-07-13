import re

FIELD_REF_PATTERN = re.compile(r"\{\{\{(.+?)\}\}\}")


class FakeSpectrum:
    def __init__(self, metadata):
        self.metadata = metadata

    def get_metadata_value(self, key):
        return self.metadata.get(key)


def resolve(template, spectrum):
    def repl(m):
        return spectrum.get_metadata_value(m.group(1).strip()) or ""

    return FIELD_REF_PATTERN.sub(repl, template)


s = FakeSpectrum({"NAME": "Glucose", "MW": "180.16"})
print(resolve("Compound: {{{NAME}}} ({{{MW}}} g/mol)", s))
print(resolve("{{{MW}}}", s))
print(int(resolve("{{{MW}}}", s).split(".")[0]))
print(float(resolve("{{{MW}}}", s)))
