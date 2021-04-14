def PlainCoNLLReader(filepath, min_len=0, max_len=300):
    with open(filepath) as fin:
        sentence = []
        to_include = True
        for line in fin:
            line = line.strip()
            if line.startswith("#") or not len(line):
                if min_len < len(sentence) < max_len:
                    if to_include:
                        yield sentence
                sentence = []
                to_include = True
                if line.startswith("speaker: CHI"):
                    to_include = False
            else:
                line = line.strip()
                if len(line):
                    sentence.append(line)

        if min_len < len(sentence) < max_len and to_include:
            yield sentence
