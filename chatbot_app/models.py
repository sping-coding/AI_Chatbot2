from django.db import models

class Conversation(models.Model):
    user_input = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user_input}, Bot: {self.bot_response}"
