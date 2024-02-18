from EmotionCauseLink import EmotionCauseLink

class UtteranceItem:

    def __init__(self, utterance_id, utterance_text, utterance_emotion, utterance_speaker, video_name):
        self.utterance_id = utterance_id
        self.utterance_text = utterance_text
        self.utterance_emotion = utterance_emotion
        self.utterance_speaker = utterance_speaker
        self.video_name = video_name
        self.emotion_cause_links = []

    def append_emotion_cause_link(self, emotion_cause_link: EmotionCauseLink):
        self.emotion_cause_links.append(emotion_cause_link)