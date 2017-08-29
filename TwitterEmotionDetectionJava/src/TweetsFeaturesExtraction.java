import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

public class TweetsFeaturesExtraction {
    private HashMap<String, Dimensions> lexicon;
    private List<Tweet> tweets;
    /*
        Intensifiers:
        deeply , always, absolutely, completely, extremely, highly, rather, really, so, too, totally, utterly, very, at all, extraordinarily

        Mitigators:
        fairly, somewhat, rather, quite, lack, least, less, slightly, a little, a little bit, a bit, just a bit

        Conjunctive adverbs:
        however, but, although, anyway, besides, later, instead, next, still, also
     */

    public TweetsFeaturesExtraction(File fileForLexicon, File fileForDataset) {
        lexicon = new HashMap<>();
        tweets = new ArrayList<>();
        loadLexicon(fileForLexicon);
        loadTweets(fileForDataset);
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
        // TODO: save file as csv
    }

    private void changeValence(List<TypedDependency> tdl, Tweet t) {
        // first rule - Negation
        negationValenceModification(tdl, t);
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
            if (rel.getLongName().equals("negation modifier") || negatives.contains(dependent)) {
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

    private void changeValenceNegationRule(WordStruct word) {
        double valence = word.getValence();
        valence *= -1;
        word.setValence(valence);
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
            arousal = lexicon.get(w).getArousal() - 4;
        }
        word.setValence(valence);
        word.setArousal(arousal);
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
