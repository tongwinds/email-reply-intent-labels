# email-reply-intent-labels
Email reply intent labels

Labels 840 sales email replies by buying intent (High, Medium, Low, None, Auto-reply) using a free OpenRouter model. Each label comes with a short reason.

How the OpenRouter key is set up
The key lives in a local .env file, never in the code.
label_replies.py reads OPENROUTER_API_KEY from .env. No key is written in any script.
.env is listed in .gitignore, so it is not uploaded to this repo.
.env.example shows the expected format with a placeholder value.

To run it yourself:

Copy .env.example to .env and paste your own OpenRouter key.
Install the dependency: pip install requests
Test on 20 replies: python label_replies.py --input replies.csv --limit 20
Run everything: python label_replies.py --input replies.csv
Retry failures: python label_replies.py --input replies.csv --retry-failed

Default model: inclusionai/ling-3.2-flash:free. Change it with --model.

Files
label_replies.py: labeling script
labeling_guide.md: what each label means, with examples
labeled_replies.csv: output with label and reason per reply

Built with Google Colab and a Python script instead of Claude Code.
