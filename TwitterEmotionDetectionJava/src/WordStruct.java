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

    public Dimensions getDimensions() {
        return this.dimensions;
    }

    public String getWord() {
        return word;
    }

    public String getWordRoot() {
        return wordRoot;
    }

    @Override
    public String toString() {
        return String.format("Word: %s\nLemma: %s\nValence: %.2f\nArousal: %.2f\n",
                word, wordRoot, dimensions.getValence(), dimensions.getArousal());
    }
}
