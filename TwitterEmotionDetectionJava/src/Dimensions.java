public class Dimensions implements Comparable<Dimensions> {
    private double valence;
    private double arousal;

    public Dimensions(double valence, double arousal) {
        this.valence = valence;
        this.arousal = arousal;
    }

    public double getValence() {
        return valence;
    }

    public void setValence(double valence) {
        this.valence = valence;
    }

    public double getArousal() {
        return arousal;
    }

    public void setArousal(double arousal) {
        this.arousal = arousal;
    }

    @Override
    public int compareTo(Dimensions o) {
        return Double.compare(this.valence, o.valence);
    }
}
