import java.util.List;

public class Tweet {
    private int id;
    private List<WordStruct> words;
    private String emotion;

    public Tweet(int id, List<WordStruct> words, String emotion) {
        this.id = id;
        this.emotion = emotion;
        this.words = words;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public List<WordStruct> getWords() {
        return words;
    }

    public void setWords(List<WordStruct> words) {
        this.words = words;
    }

    public String getEmotion() {
        return emotion;
    }
}
