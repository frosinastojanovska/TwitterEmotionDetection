import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.*;

import java.io.*;
import java.util.*;

public class TweetsFeaturesExtraction {
    private HashMap<String, Dimensions> lexicon;
    private List<Tweet> tweets;
    private HashSet<String> intensifiers;
    private HashSet<String> mitigators;
    private HashSet<String> conjunctiveAdverbs;

    public TweetsFeaturesExtraction(File fileForLexicon, File fileForDataset) {
        lexicon = new HashMap<>();
        tweets = new ArrayList<>();
        loadSets();
        loadLexicon(fileForLexicon);
        loadTweets(fileForDataset);
    }

    public Tweet calculateValenceOfWordsOneTweet(int tweetId) {
        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
        for (Tweet t : tweets) {
            if (t.getId() == tweetId) {
                String[] words = new String[t.getWords().size()];
                int i = 0;
                for (WordStruct w : t.getWords()) {
                    getValence(w);
                    words[i] = w.getWord();
                    i++;
                }
                List<TypedDependency> tdl = getDependencies(words, lp);
                changeValence(tdl, t);
                return t;
            }
        }
        return null;
    }

    public void calculateValenceOfWords() {
        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
        for (Tweet t : tweets) {
            String[] words = new String[t.getWords().size()];
            int i = 0;
            for (WordStruct w : t.getWords()) {
                getValence(w);
                words[i] = w.getWord();
                i++;
            }
            List<TypedDependency> tdl = getDependencies(words, lp);
            changeValence(tdl, t);
        }
    }

    public void saveFeaturesToFile(File output) {
        BufferedWriter writer = null;
        try {
            writer = new BufferedWriter(new FileWriter(output));
            writer.write("tweet_id,valMax1,arlMax1,valMax2,arlMax2,valMax3,arlMax3,valMax4,arlMax4," +
                    "valMax5,arlMax5,valMin1,arlMin1,valMin2,arlMin2,valMin3,arlMin3,valMin4,arlMin4," +
                    "valMin5,arlMin5,emotion\n");
            for (Tweet t : tweets) {
                List<Dimensions> positiveValences = new ArrayList<>();
                List<Dimensions> negativeValences = new ArrayList<>();
                for (WordStruct w : t.getWords()) {
                    double valence = w.getValence();
                    Dimensions dimensions = w.getDimensions();
                    if (valence < 0)
                        negativeValences.add(dimensions);
                    else if (valence > 0)
                        positiveValences.add(dimensions);
                    else {
                        positiveValences.add(dimensions);
                        negativeValences.add(dimensions);
                    }
                }
                while (positiveValences.size() < 5)
                    positiveValences.add(new Dimensions(0.0, 0.0));
                while (negativeValences.size() < 5)
                    negativeValences.add(new Dimensions(0.0, 0.0));

                Collections.sort(positiveValences);
                Collections.reverse(positiveValences);
                Collections.sort(negativeValences);

                String positive = mergeFeatures(positiveValences);
                String negative = mergeFeatures(negativeValences);

                String line = String.format("%d,%s%s%s\n", t.getId(), positive, negative, t.getEmotion());
                writer.write(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private String mergeFeatures(List<Dimensions> dimensions) {
        StringBuilder builder = new StringBuilder();
        String SEPARATOR = ",";
        for (Dimensions d : dimensions.subList(0, 5)) {
            builder.append(d.getValence());
            builder.append(SEPARATOR);
            builder.append(d.getArousal());
            builder.append(SEPARATOR);
        }
        return builder.toString();
    }

    private void changeValence(List<TypedDependency> tdl, Tweet t) {
        // first rule - Negation
        negationValenceModification(tdl, t);
        // second rule - Intensifiers
        intensifiersValenceModification(tdl, t);
        // third rule - Mitigators
        mitigatorsValenceModification(tdl, t);
        // forth rule - Negative words shifters
        negativeWordsValenceModification(tdl, t);
        // fifth rule - Conjunctive Adverbs (connectors)
        conjunctiveAdverbsValenceModification(t);
    }

    private void negationValenceModification(List<TypedDependency> tdl, Tweet tweet) {
        HashSet<String> negatives = new HashSet<>();
        negatives.add("nobody");
        negatives.add("never");
        negatives.add("none");
        negatives.add("nowhere");
        negatives.add("nothing");
        negatives.add("neither");

        for (TypedDependency td : tdl) {
            GrammaticalRelation rel = td.reln();
            String dependent = td.dep().word().toLowerCase();
            if (rel.getLongName().equals("negation modifier") ||
                    (negatives.contains(dependent) && td.gov().index() != 0)) {
                int governor = td.gov().index();
                for (TypedDependency second_td : tdl) {
                    int gov = second_td.gov().index();
                    int dep = second_td.dep().index();
                    boolean flag = second_td.dep().word().toLowerCase().equals(dependent);
                    if (gov == governor && !flag) {
                        WordStruct wordForChanging = tweet.getWordAt(dep - 1);
                        changeValenceNegationRule(wordForChanging);
                    }
                }
                WordStruct wordForChanging = tweet.getWordAt(governor - 1);
                changeValenceNegationRule(wordForChanging);
            }
        }
    }

    private void intensifiersValenceModification(List<TypedDependency> tdl, Tweet tweet) {

        for (TypedDependency td : tdl) {
            String rel = td.reln().getLongName();
            String dependent = td.dep().word().toLowerCase();
            if (intensifiers.contains(dependent) && !rel.equals("root")) {
                int governor = td.gov().index();
                for (TypedDependency second_td : tdl) {
                    int gov = second_td.gov().index();
                    int dep = second_td.dep().index();
                    boolean flag = second_td.dep().word().toLowerCase().equals(dependent);
                    if (gov == governor && !flag) {
                        WordStruct wordForChanging = tweet.getWordAt(dep - 1);
                        changeValenceIntensifiersRule(wordForChanging);
                    }
                }
                WordStruct wordForChanging = tweet.getWordAt(governor - 1);
                changeValenceIntensifiersRule(wordForChanging);
            }
        }
    }

    private void mitigatorsValenceModification(List<TypedDependency> tdl, Tweet tweet) {

        for (TypedDependency td : tdl) {
            String rel = td.reln().getLongName();
            String dependent = td.dep().word().toLowerCase();
            if (mitigators.contains(dependent) && !rel.equals("root")) {
                int governor = td.gov().index();
                for (TypedDependency second_td : tdl) {
                    int gov = second_td.gov().index();
                    int dep = second_td.dep().index();
                    boolean flag = second_td.dep().word().toLowerCase().equals(dependent);
                    if (gov == governor && !flag) {
                        WordStruct wordForChanging = tweet.getWordAt(dep - 1);
                        changeValenceMitigatorsRule(wordForChanging);
                    }
                }
                WordStruct wordForChanging = tweet.getWordAt(governor - 1);
                changeValenceMitigatorsRule(wordForChanging);
            }
        }
    }

    private void conjunctiveAdverbsValenceModification(Tweet tweet) {
        boolean flag = false;
        for (WordStruct word : tweet.getWords()) {
            if (conjunctiveAdverbs.contains(word.getWord())) {
                flag = true;
                break;
            }
        }
        if (flag) {
            for (WordStruct word : tweet.getWords()) {
                changeValenceConjunctiveAdverbsRule(word);
            }
        }
    }

    private void negativeWordsValenceModification(List<TypedDependency> tdl, Tweet tweet) {
        for (TypedDependency td : tdl) {
            double dependentValence = tweet.getWordAt(td.dep().index() - 1).getValence();
            int dependent = td.dep().index();
            int governor = td.gov().index();
            if (governor == 0)
                continue;
            double governorValence = tweet.getWordAt(td.gov().index() - 1).getValence();
            if (dependentValence < 0 && governorValence > 0) {
                for (TypedDependency second_td : tdl) {
                    int gov = second_td.gov().index();
                    int dep = second_td.dep().index();
                    boolean flag = second_td.dep().index() == dependent;
                    if (gov == governor && !flag) {
                        WordStruct wordForChanging = tweet.getWordAt(dep - 1);
                        changeValenceNegativeWordsRule(wordForChanging);
                    }
                }
                WordStruct wordForChanging = tweet.getWordAt(governor - 1);
                changeValenceNegativeWordsRule(wordForChanging);
            }
        }
    }

    private void changeValenceNegationRule(WordStruct word) {
        double valence = word.getValence();
        valence *= -1;
        word.setValence(valence);
    }

    private void changeValenceIntensifiersRule(WordStruct word) {
        double valence = word.getValence();
        valence *= 1.5;
        word.setValence(valence);
    }

    private void changeValenceMitigatorsRule(WordStruct word) {
        double valence = word.getValence();
        valence *= 0.5;
        word.setValence(valence);
    }

    private void changeValenceConjunctiveAdverbsRule(WordStruct word) {
        double valence = 0;
        word.setValence(valence);
    }

    private void changeValenceNegativeWordsRule(WordStruct word) {
        double valence = word.getValence();
        if (valence > 0) {
            valence *= -1;
            word.setValence(valence);
        }
    }

    private List<TypedDependency> getDependencies(String[] tweetWords, LexicalizedParser lp) {
        List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(tweetWords);
        Tree parse = lp.apply(rawWords);
        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // PennTreebankLanguagePack for English
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        return gs.typedDependenciesCCprocessed();
    }

    private void getValence(WordStruct word) {
        double valence = 0;
        double arousal = 0;
        String w = word.getWordRoot().toLowerCase();
        if (lexicon.containsKey(w)) {
            valence = lexicon.get(w).getValence() - 4.5;
            arousal = lexicon.get(w).getArousal() - 4.5;
        }
        word.setValence(valence);
        word.setArousal(arousal);
    }

    private void loadSets() {
        intensifiers = new HashSet<>();
        mitigators = new HashSet<>();
        conjunctiveAdverbs = new HashSet<>();

        intensifiers.add("absolutely");
        intensifiers.add("always");
        intensifiers.add("amazingly");
        intensifiers.add("completely");
        intensifiers.add("deeply");
        intensifiers.add("exceptionally");
        intensifiers.add("extraordinary");
        intensifiers.add("extremely");
        intensifiers.add("highly");
        intensifiers.add("incredibly");
        intensifiers.add("really");
        intensifiers.add("remarkably");
        intensifiers.add("so");
        intensifiers.add("super");
        intensifiers.add("too");
        intensifiers.add("totally");
        intensifiers.add("utterly");
        intensifiers.add("very");

        mitigators.add("fairly");
        mitigators.add("rather");
        mitigators.add("quite");
        mitigators.add("lack");
        mitigators.add("least");
        mitigators.add("less");
        mitigators.add("slightly");

        conjunctiveAdverbs.add("however");
        conjunctiveAdverbs.add("but");
        conjunctiveAdverbs.add("although");
        conjunctiveAdverbs.add("anyway");
        conjunctiveAdverbs.add("besides");
        conjunctiveAdverbs.add("later");
        conjunctiveAdverbs.add("instead");
        conjunctiveAdverbs.add("next");
        conjunctiveAdverbs.add("still");
        conjunctiveAdverbs.add("also");
    }

    private void loadLexicon(File file) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            // read header
            String header = reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String parts[] = line.split(",");
                String word = parts[1];
                float valence = Float.parseFloat(parts[2]);
                float arousal = Float.parseFloat(parts[5]);
                Dimensions dimensions = new Dimensions(valence, arousal);
                lexicon.put(word, dimensions);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void loadTweets(File file) {
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            // read header
            String header = reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String parts[] = line.split(",");
                int id = Integer.parseInt(parts[0]);
                String emotion = parts[1];
                String[] words = parts[2].split(";");
                List<WordStruct> list = new ArrayList<>();
                for (String word : words) {
                    String[] wordParts = word.split("//");
                    WordStruct w = new WordStruct(wordParts[0], wordParts[1]);
                    list.add(w);
                }
                Tweet tweet = new Tweet(id, list, emotion);
                tweets.add(tweet);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
