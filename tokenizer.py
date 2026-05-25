import re

class Simpletokenizer:

    def __init__(self, vocab):

        self.str_to_init = vocab

        self.init_to_str = {
            i: s for s, i in vocab.items()
        }

    def encode(self, text):

        preprocessed = re.split(
            r'([,.:;?_!"()\']|--|\s)',
            text
        )

        new_list = []

        for item in preprocessed:

            cleaned_item = item.strip()

            if cleaned_item != "":

                new_list.append(cleaned_item)

        token_ids = []

        for token in new_list:

            if token in self.str_to_init:

                token_ids.append(
                    self.str_to_init[token]
                )

            else:

                token_ids.append(
                    self.str_to_init["<unk>"]
                )

        return token_ids

    def decode(self, token_ids):

        text = " ".join(
            [
                self.init_to_str[i]
                for i in token_ids
            ]
        )

        text = re.sub(
            r'\s+([,.:;?!"()\'])',
            r'\1',
            text
        )

        return text