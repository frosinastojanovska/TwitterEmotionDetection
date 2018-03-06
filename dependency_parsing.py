from nltk.parse.stanford import StanfordDependencyParser


class DependencyParsing:
    """
    Stanford dependency parsing
    """

    def __init__(self, path_to_jar, path_to_models_jar):
        self.dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar,
                                                          path_to_models_jar=path_to_models_jar)

    def parse_sentences(self, sentences):
        """ Dependency parsing of list of tokenized sentences using the stanford parser

        :param sentences: List of sentences. Each sentence is a list of tokens.
        :type sentences: list(list(str))
        :return: iterator of DependencyGraph objects
        :rtype: iterator
        """
        result = self.dependency_parser.parse_sents(sentences)
        return result.__next__()

    def parse_sentence(self, sentence):
        """ Dependency parsing of a tokenized sentence using the stanford parser

        :param sentence: sentence as a list of tokens.
        :type sentence: list(str)
        :return: DependencyGraph object
        :rtype: nltk.DependencyGraph
        """
        result = self.dependency_parser.parse(sentence)
        return result.__next__()
