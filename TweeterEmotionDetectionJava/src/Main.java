import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.process.Morphology;
import edu.stanford.nlp.ling.Word;

import java.io.*;
import java.util.*;

public class Main {

    private static HashMap<String, Dimensions> lexicon = new HashMap<>();

    public static void main(String[] args) {
        Morphology m = new Morphology();
        System.out.println(m.stem(new Word("seeing")));
        System.out.println(Morphology.stemStatic("seeing", "VB"));
        loadLexicon();
        getValence(new String[]{"I", "genuinely", "think", "I", "have", "anger", "issue"});
        String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
        LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
        demoAPI(lp);

    }

    public static void changeValence(){

    }

    public static void getValence(String tokens[]){
        float sumValence = 0;
        float sumArousal = 0;
        for(String word : tokens){
            float valence = 0;
            float arousal = 0;
            word = word.toLowerCase();
            if(lexicon.containsKey(word)) {
                valence = lexicon.get(word).valence;
                arousal = lexicon.get(word).arousal;
            }
            sumValence += valence;
            sumArousal += arousal;
            System.out.println("Word-> " + word + " valence-> " + valence);
            System.out.println("Word-> " + word + " arousal-> " + arousal);
        }
        System.out.println("Sum valence: " + sumValence);
        System.out.println("Sum arousal: " + sumArousal);
    }

    public static void loadLexicon(){
        File file = new File("Data/Ratings_Warriner_et_al.csv");
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

    public static void demoAPI(LexicalizedParser lp) {
        // This option shows parsing a list of correctly tokenized words
        String[] sent = {"Happiness", "is", "always", "there", ".", "You", "just", "have", "to", "choose", "to", "see", "it", "."};
        List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(sent);
        Tree parse = lp.apply(rawWords);
        parse.pennPrint();
        System.out.println();

        // This option shows loading and using an explicit tokenizer
        String sent2 = "I don't think that my life is a failure.";
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
