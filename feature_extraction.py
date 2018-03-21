from dependency_parsing import DependencyParsing
from itertools import chain


class FeatureExtractionContextValenceShifting:
    """
    Class containing the logic for contextual shifting of the valence
    """

    def __init__(self, path_to_jar, path_to_models_jar, lexicon, model=None):
        """

        :param path_to_jar: file path to the jar file of the stanford parser
        :type path_to_jar: str
        :param path_to_models_jar: file path to the jar file of the stanford parser model
        :type path_to_models_jar: str
        :param lexicon: data frame containing the lexicon with columns word and valence.
        :type lexicon: pandas.DataFrame
        :param model: Gensim word2vec model (only if its needed, default is None)
        :type model: gensim.models.KeyedVectors
        """
        self.parser = DependencyParsing(path_to_jar, path_to_models_jar)
        self.negatives = {'nobody', 'never', 'none', 'nowhere', 'nothing', 'neither'}
        self.intensifiers = {'absolutely', 'always', 'amazingly', 'completely', 'deeply',
                             'exceptionally', 'extraordinary', 'extremely', 'highly',
                             'incredibly', 'really', 'remarkably', 'so', 'super', 'too',
                             'totally', 'utterly', 'very'}
        self.mitigators = {'fairly', 'rather', 'quite', 'lack', 'least', 'less', 'slightly'}
        self.conjunctive_adverbs = {'however', 'but', 'although', 'anyway', 'besides',
                                    'later', 'instead', 'next', 'still', 'also'}
        self.lexicon = lexicon
        self.model = model

    def get_initial_valences(self, df):
        """ Adds additional column with valences for every word (token/lemma)

        :param df: data frame containing the dataset with arbitrary columns tokens and lemmas.
        :type df: pandas.DataFrame
        :return: data frame with additional column valences for the valences of each lemma
        :rtype: pandas.DataFrame
        """
        df['valences'] = ''
        for index, row in df.iterrows():
            valences = []
            for sent in row.lemmas:
                valences.append(self.get_initial_valences_sentence(sent))
            if index % 100 == 0:
                print(index)
            df.set_value(index=index, col='valences', value=valences)

        return df

    def get_initial_valences_sentence(self, sentence):
        """ Get the valences from sentence lemmas

        :param sentence: data frame containing the dataset with arbitrary columns tokens and lemmas.
        :type sentence: list(str)
        :return: list of valences for each lemma in the sentence
        :rtype: list(float)
        """
        if self.model is None:
            valences = [self.get_valence(lemma)
                        if lemma in self.lexicon.word.values.tolist() else 0
                        for lemma in sentence]
        else:
            valences = [0] * len(sentence)
            for lemma, i in zip(sentence, range(len(sentence))):
                if lemma in self.lexicon.word.values.tolist():
                    valences[i] = self.get_valence(lemma)
                else:
                    if lemma not in self.model.vocab:
                        continue
                    for sim_lemma in self.model.most_similar(lemma, topn=5):
                        if sim_lemma in self.lexicon.word.values.tolist():
                            valences[i] = self.get_valence(sim_lemma)
                            break
        return valences

    def get_valence(self, lemma):
        return self.lexicon.loc[self.lexicon['word'] == lemma].valence.values[0] - 4.5

    def change_valence_dataset(self, df):
        """ Change the initial valences in the dataset with the contextual shifters rules

        :param df: data frame containing the dataset
        :type df: pandas.DataFrame
        :return: modified valences in the dataset with contextual shifters rules
        :rtype: pandas.DataFrame
        """
        for index, row in df.iterrows():
            valence = []
            for sentence, valences in zip(row.tokens, row.valences):
                modified_valences = self.change_valence_sentence(sentence, valences)
                valence.append(modified_valences)

            df.set_value(index=index, col='valences', value=valence)

    def change_valence_sentence(self, sentence, valences):
        """ Change the initial valences of the sentence with the contextual shifters rules

        :param sentence: tokenized sentence
        :type sentence: list(str)
        :param valences: list of the valences of the tokens in the sentence
        :type valences: list(float)
        :return: modified valences
        :rtype: list(float)
        """
        dg = self.parser.parse_sentence(sentence)
        # first rule - Negation
        modified_valences = self.negation_valence_modification(dg, valences)
        # second rule - Intensifiers
        modified_valences = self.intensifiers_valence_modification(dg, modified_valences)
        # third rule - Mitigators
        modified_valences = self.mitigators_valence_modification(dg, modified_valences)
        # forth rule - Negative words shifters
        modified_valences = self.negative_words_valence_modification(dg, modified_valences)
        # fifth rule - Conjunctive Adverbs (connectors)
        modified_valences = self.conjunctive_adverbs_valence_modification(dg, modified_valences)

        return modified_valences

    def negation_valence_modification(self, dg, valences):
        """ Change the valences with the negation rule

        :param dg: DependencyGraph object
        :type dg: nltk.DependencyGraph
        :param valences: list of valences
        :type valences: list(float)
        :return: modified valence values
        :rtype: list(float)
        """
        for node in dg.nodes.values():
            if node['rel'] == 'neg' or str(node['word']).lower() in self.negatives:
                address = node['address']
                for n in dg.nodes.values():
                    if n['address'] == address:
                        continue
                    if address in list(chain.from_iterable(n['deps'].values())):
                        valences[n['address'] - 1] = self.change_valence_negation_rule(valences[n['address'] - 1])

        return valences

    def intensifiers_valence_modification(self, dg, valences):
        """ Change the valences with the intensifiers rule

        :param dg: DependencyGraph object
        :type dg: nltk.DependencyGraph
        :param valences: list of valences
        :type valences: list(float)
        :return: modified valence values
        :rtype: list(float)
        """
        for node in dg.nodes.values():
            if str(node['word']).lower() in self.intensifiers:
                address = node['address']
                for n in dg.nodes.values():
                    if n['address'] == address:
                        continue
                    if address in list(chain.from_iterable(n['deps'].values())):
                        valences[n['address'] - 1] = self.change_valence_intensifiers_rule(valences[n['address'] - 1])

        return valences

    def mitigators_valence_modification(self, dg, valences):
        """ Change the valences with the mitigators rule

        :param dg: DependencyGraph object
        :type dg: nltk.DependencyGraph
        :param valences: list of valences
        :type valences: list(float)
        :return: modified valence values
        :rtype: list(float)
        """
        for node in dg.nodes.values():
            if str(node['word']).lower() in self.mitigators:
                address = node['address']
                for n in dg.nodes.values():
                    if n['address'] == address:
                        continue
                    if address in list(chain.from_iterable(n['deps'].values())):
                        valences[n['address'] - 1] = self.change_valence_mitigators_rule(valences[n['address'] - 1])

        return valences

    def negative_words_valence_modification(self, dg, valences):
        """Change the valences with the negative words rule

        :param dg: DependencyGraph object
        :type dg: nltk.DependencyGraph
        :param valences: list of valences
        :type valences: list(float)
        :return: modified valence values
        :rtype: list(float)
        """
        for node in dg.nodes.values():
            if valences[node['address'] - 1] < 0:
                # if the word is negative, change the dependencies
                for dep in list(chain.from_iterable(node['deps'].values())):
                    if valences[dep - 1] > 0:
                        valences[dep - 1] = \
                            self.change_valence_negative_words_rule(valences[dep - 1])
                        # change also the direct dependencies of the affected word
                        children = list(chain.from_iterable([x['deps'].values()
                                                             for x in list(dg.nodes.values())
                                                             if x['address'] == dep][0]))
                        for d in children:
                            if valences[d - 1] > 0:
                                valences[d - 1] = \
                                    self.change_valence_negative_words_rule(valences[d - 1])
            else:
                # check if some of the dependencies is a negative word and change the valence in that case
                for dep in list(chain.from_iterable(node['deps'].values())):
                    if valences[dep - 1] < 0:
                        valences[node['address'] - 1] = \
                            self.change_valence_negative_words_rule(valences[node['address'] - 1])
                        break
        return valences

    def conjunctive_adverbs_valence_modification(self, dg, valences):
        """Change the valences with the conjunctive adverbs (connectors) rule

        :param dg: DependencyGraph object
        :type dg: nltk.DependencyGraph
        :param valences: list of valences
        :type valences: list(float)
        :return: modified valence values
        :rtype: list(float)
        """
        for node in dg.nodes.values():
            if str(node['word']).lower() in self.conjunctive_adverbs:
                address = node['address']
                for n in dg.nodes.values():
                    if n['address'] == address:
                        continue
                    if address in list(chain.from_iterable(n['deps'].values())):
                        valences[n['address'] - 1] = \
                            self.change_valence_conjunctive_adverbs_rule(valences[n['address'] - 1])

        return valences

    @staticmethod
    def change_valence_negation_rule(valence):
        """ Negation rule for valence change

        :param valence: the word valence value
        :type valence: float
        :return: modified valence value
        :rtype: float
        """
        valence *= -1
        return valence

    @staticmethod
    def change_valence_intensifiers_rule(valence):
        """ Intensifiers rule for valence change

        :param valence: the word valence value
        :type valence: float
        :return: modified valence value
        :rtype: float
        """
        valence *= 1.5
        return valence

    @staticmethod
    def change_valence_mitigators_rule(valence):
        """ Mitigators rule for valence change

        :param valence: the word valence value
        :type valence: float
        :return: modified valence value
        :rtype: float
        """
        valence *= 0.5
        return valence

    @staticmethod
    def change_valence_conjunctive_adverbs_rule(valence):
        """ Conjunctive adverbs rule for valence change

        :param valence: the word valence value
        :type valence: float
        :return: modified valence value
        :rtype: float
        """
        valence *= 0
        return valence

    @staticmethod
    def change_valence_negative_words_rule(valence):
        """ Negative words rule for valence change

        :param valence: the word valence value
        :type valence: float
        :return: modified valence value
        :rtype: float
        """
        if valence > 0:
            valence *= -1
        return valence
