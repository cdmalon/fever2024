import spacy

class NER_Retriever():

    def __init__(self):
        self.interesting_ner = ["PERSON", "FAC", "ORG", "GPE", "LOC",
                                "PRODUCT", "EVENT", "WORK_OF_ART", "PER",
                                "LOC", "MISC"]
        self.months = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November",
                       "December"]
        self.addresses = ["President", "Chairman", "Chairwoman", "Prime",
                          "Minister"]

        self.nlp = spacy.load("en_core_web_lg")

    def make_lookup(self, tok, i, j):
        lookup = tok[i].text
        for k in range(i+1, j+1):
          if(tok[k].text == ","):
            lookup = lookup + ","
          else:
            lookup = lookup + " " + tok[k].text
        return lookup

    def lookups(self, claim_text):
        tok = self.nlp(claim_text)
        claim = " ".join(map(lambda x: x.text, tok))

        uppers = []
        ner = []
        for token in tok:
          uppers.append(token.text[0].isupper())
          ner.append(token.ent_type_)

        lookups = []
        i = 0
        while i < len(tok):
          stop_i = i
          lookup = None
          # First word capitalization starts a lookup only if next word is
          # capitalized or it has an interesting NER tag
          # But any later capitalized word can start a lookup
          if((i == 0 and (ner[0] in self.interesting_ner or
                          (1 < len(tok) and uppers[1]))) or
             (i > 0 and uppers[i])):
            stopped = False
            for j in range(i, len(tok)):
              # j: where this named entity or capitalized phrase might end
              if(stopped == False):
                # Spans can only pass through NE, capitalized, or commas
                if(tok[j].text == "," or (ner[j] in self.interesting_ner) or uppers[j]):
                  # Lookup can end where next word not capitalized or is comma
                  # But keep trying to extend a span beyond a comma
                  if(tok[j].text != "," and not (j+1 < len(uppers) and uppers[j+1])):
                    # Form the candidate Wikipedia page title
                    lookup = self.make_lookup(tok, i, j)
                    # Don't retrieve Wikipedia pages about addresses and months
                    if(not(lookup in self.addresses or lookup in self.months)):
                      stop_i = j

                else:
                  stopped = True  # can't join consecutive words into lookup key
                  if lookup is not None:
                    lookups.append(lookup)

          i = stop_i + 1

        return lookups

