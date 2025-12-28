import React, { useState, useCallback } from 'react';
import clsx from 'clsx';

const CopyButton = ({ code, className }) => {
  const [copied, setCopied] = useState(false);

  const handleCopyCode = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);

      setTimeout(() => {
        setCopied(false);
      }, 2000);
    } catch (err) {
      console.error('Failed to copy code: ', err);
    }
  }, [code]);

  return (
    <button
      className={clsx(
        'clean-btn',
        'copy-button',
        copied ? 'copy-button--success' : '',
        className
      )}
      onClick={handleCopyCode}
      title="Copy code to clipboard"
      type="button"
      aria-label={copied ? 'Copied' : 'Copy code to clipboard'}
    >
      {copied ? (
        <span className="copy-button__icon copy-button__icon--success">
          ✓
        </span>
      ) : (
        <span className="copy-button__icon">📋</span>
      )}
      <span className="copy-button__label">
        {copied ? 'Copied!' : 'Copy'}
      </span>
    </button>
  );
};

export default CopyButton;