import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

type Props = {
  disabled: boolean;
  onSend: (text: string) => void;
};

const MAX_HEIGHT = 200;

export function Composer({ disabled, onSend }: Props) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow the field with its content, up to MAX_HEIGHT.
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, MAX_HEIGHT)}px`;
  }, [text]);

  useEffect(() => {
    if (!disabled) textareaRef.current?.focus();
  }, [disabled]);

  function submit() {
    if (disabled || !text.trim()) return;
    onSend(text);
    setText("");
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    submit();
  }

  // Enter sends, Shift+Enter adds a new line.
  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Задайте вопрос по документам…"
        aria-label="Вопрос"
        rows={1}
        maxLength={4000}
      />
      <button type="submit" className="send-button" disabled={disabled || !text.trim()}>
        Отправить
      </button>
    </form>
  );
}
