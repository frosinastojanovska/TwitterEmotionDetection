import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.*;

import java.io.*;
import java.util.*;

public class Main {

    private static HashMap<String, Dimensions> lexicon = new HashMap<>();
    private static List<Tweet> tweets = new ArrayList<>();

    /*
        Intensifiers:
        deeply , always, absolutely, completely, extremely, highly, rather, really, so, too, totally, utterly, very, at all, extraordinarily

        Mitigators:
        fairly, somewhat, rather, quite, lack, least, less, slightly, a little, a little bit, a bit, just a bit

        Conjunctive adverbs:
        however, but, although, anyway, besides, later, instead, next, still, also
     */

    public static void main(String[] args) {
        File file = new File("Data/Ratings_Warriner_et_al.csv");
        loadLexicon(file);
        file = new File("Data/full_dataset_tokens.csv");
        loadTweets(file);
//        getValence(new String[]{"I", "don't", "know", "why", "people", "are", "so", "angry"});
        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
        for (Tweet t : tweets){
            String[] words = new String[t.getWords().size()];
            int i = 0;
            for (WordStruct w : t.getWords()) {
                getValence(w);
                words[i] = w.getWord();
                i++;
            }
            List<TypedDependency> tdl = getDependencies(words, lp);
            System.out.println(tdl.get(0));
            for(TypedDependency td : tdl){
                String[] parts = td.toString().split("\\(");
                String relation = parts[0];
                System.out.println(relation);
            }
        }
        demoAPI(lp);
    }

    private static void loadTweets(File file){
        BufferedReader reader = null;
        try{
            reader = new BufferedReader(new FileReader(file));
            // read header
            String line = reader.readLine();
            while((line = reader.readLine()) != null){
                String parts[] = line.split(",");
                int id = Integer.parseInt(parts[0]);
                String emotion = parts[1];
                String[] words = parts[2].split(";");
                List<WordStruct> list = new ArrayList<>();
                for(String word : words){
                    String[] wordParts = word.split("//");
                    WordStruct w = new WordStruct(wordParts[0], wordParts[1]);
                    list.add(w);
                }
                Tweet tweet = new Tweet(id, list, emotion);
                tweets.add(tweet);
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
        finally {
            if(reader != null) {
                try {
                    reader.close();
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
    }

    public static void changeValence(){

    }

    public static void getValence(WordStruct word){
            double valence = 0;
            double arousal = 0;
            String w = word.getWordRoot().toLowerCase();
            if(lexicon.containsKey(w)) {
                valence = lexicon.get(w).getValence() - 4.5;
                arousal = lexicon.get(w).getArousal() - 4;
            }
            word.setValence(valence);
            word.setArousal(arousal);
    }

    public static void loadLexicon(File file){
        BufferedReader reader = null;
        try{
            reader = new BufferedReader(new FileReader(file));
            // read header
            String line = reader.readLine();
            while((line = reader.readLine()) != null){
                String parts[] = line.split(",");
                String word = parts[1];
                float valence = Float.parseFloat(parts[2]);
                float arousal = Float.parseFloat(parts[5]);
                Dimensions dimensions = new Dimensions(valence, arousal);
                lexicon.put(word, dimensions);
            }
        }
        catch (IOException e){
            e.printStackTrace();
        }
        finally {
            if(reader != null) {
                try {
                    reader.close();
                }
                catch (IOException e){
                    e.printStackTrace();
                }
            }
        }
    }

    public static List<TypedDependency> getDependencies(String[] tweetWords, LexicalizedParser lp){
        List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(tweetWords);
        Tree parse = lp.apply(rawWords);
//        parse.pennPrint();
//        System.out.println();
        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // PennTreebankLanguagePack for English
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
//        System.out.println(tdl);
        return tdl;
    }

    public static void demoAPI(LexicalizedParser lp) {
        // This option shows parsing a list of correctly tokenized words
        String[] sent = {"Happiness", "is", "always", "there", ".", "You", "just", "have", "to", "choose", "to", "see", "it", "."};
        List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(sent);
        Tree parse = lp.apply(rawWords);
        parse.pennPrint();
        System.out.println();

//        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // PennTreebankLanguagePack for English
//        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
//        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
//        List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
//        System.out.println(tdl);

        // This option shows loading and using an explicit tokenizer
        String sent2 = "Happiness is always there. You just have to choose to see it.";
        TokenizerFactory<CoreLabel> tokenizerFactory =
                PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
        Tokenizer<CoreLabel> tok =
                tokenizerFactory.getTokenizer(new StringReader(sent2));
        List<CoreLabel> rawWords2 = tok.tokenize();
        parse = lp.apply(rawWords2);

        TreebankLanguagePack tlp = lp.treebankLanguagePack(); // PennTreebankLanguagePack for English
        GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
        GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
        List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
        System.out.println(tdl);
        System.out.println();

        // You can also use a TreePrint object to print trees and dependencies
        TreePrint tp = new TreePrint("penn,typedDependenciesCollapsed");
        tp.printTree(parse);
    }
}
