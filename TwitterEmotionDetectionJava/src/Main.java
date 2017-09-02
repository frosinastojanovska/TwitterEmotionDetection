import java.io.File;

public class Main {

    public static void main(String[] args) {
        File file1 = new File("Data/Ratings_Warriner_et_al.csv");
        File file2 = new File("Data/merged_datasets_tokens.csv");
        TweetsFeaturesExtraction featuresExtraction = new TweetsFeaturesExtraction(file1, file2);
        // featuresExtraction.calculateValenceOfWords();
        // featuresExtraction.saveFeaturesToFile(new File("Data/merged_tweets_features.csv"));
        Tweet result = featuresExtraction.calculateValenceOfWordsOneTweet(30170);
        System.out.println(result.getWords());
    }
}
