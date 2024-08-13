_chat_template = """
{{ bos_token }}
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ 'user\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ 'assistant\\n\\n' }}
    {% endif %}
    {{ message['content'].strip() }}
    {{ '\\n' }}
{% endfor %}
"""
#DEFAULT_CHAT_TEMPLATE = "".join(line.strip() for line in _chat_template.split("\n"))

DEFAULT_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""" # llama-3 chat template
