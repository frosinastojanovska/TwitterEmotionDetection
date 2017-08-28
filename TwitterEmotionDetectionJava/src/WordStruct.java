public class WordStruct {
    private Dimensions dimensions;
    private String word;
    private String wordRoot;

    public WordStruct(String word, String wordRoot) {
        dimensions = new Dimensions(0, 0);
        this.word = word;
        this.wordRoot = wordRoot;
    }

    public double getAsousal() {
        return dimensions.getArousal();
    }

    public void setArousal(double arousal) {
        dimensions.setArousal(arousal);
    }

    public double getValence() {
        return dimensions.getValence();
    }

    public void setValence(double valence) {
        dimensions.setValence(valence);
    }

    public String getWord() {
        return word;
    }

    public String getWordRoot() {
        return wordRoot;
    }
}
