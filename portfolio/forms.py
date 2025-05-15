# portfolio/forms.py

from django import forms
from django.utils import timezone
import bleach # For sanitizing HTML content

class ContactForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-400 dark:focus:ring-blue-400 dark:focus:border-blue-400',
            'placeholder': 'Your Name'
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-400 dark:focus:ring-blue-400 dark:focus:border-blue-400',
            'placeholder': 'Your Email Address'
        })
    )
    subject = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-400 dark:focus:ring-blue-400 dark:focus:border-blue-400',
            'placeholder': 'Subject'
        })
    )
    message = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100 dark:placeholder-gray-400 dark:focus:ring-blue-400 dark:focus:border-blue-400',
            'rows': 5,
            'placeholder': 'Your Message'
        })
    )

    # Honeypot field for basic spam protection
    honeypot = forms.CharField(required=False, widget=forms.HiddenInput, label="")

    # Timestamp field for another layer of basic spam protection
    form_load_time = forms.CharField(required=False, widget=forms.HiddenInput)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial value for form_load_time when the form is instantiated (typically in the GET request)
        if not self.is_bound: # Only set on initial form load, not on POST
            self.fields['form_load_time'].initial = timezone.now().isoformat()


    def clean_message(self):
        """
        Sanitize the message content to remove potentially harmful HTML.
        Allows only a very restricted set of HTML tags.
        """
        message = self.cleaned_data.get('message', '')
        # Define allowed tags and attributes if you want to allow some HTML.
        # For a contact form message, often plain text is best.
        # bleach.clean strips disallowed tags.
        # If you want absolutely no HTML, you can use Django's strip_tags.
        # from django.utils.html import strip_tags
        # return strip_tags(message)
        
        # Example with bleach allowing only very basic formatting (bold, italic, paragraphs, line breaks)
        allowed_tags = ['p', 'br', 'b', 'strong', 'i', 'em']
        cleaned_message = bleach.clean(message, tags=allowed_tags, strip=True)
        return cleaned_message

    # Optional: Add clean_honeypot if you want to log or do something specific
    # def clean_honeypot(self):
    #     data = self.cleaned_data['honeypot']
    #     if data:
    #         # You could log this attempt
    #         # logger.warning(f"Honeypot field filled by IP: {request.META.get('REMOTE_ADDR')}")
    #         raise forms.ValidationError("Spam detected.", code='spam') # Or handle silently in the view
    #     return data

    # For reCAPTCHA (if you implement it):
    # from django_recaptcha.fields import ReCaptchaField
    # captcha = ReCaptchaField(widget=ReCaptchaV2Checkbox) # or ReCaptchaV3
