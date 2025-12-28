import React, { useState } from 'react';
import CodeBlock from '@theme/CodeBlock';
import clsx from 'clsx';

import styles from './styles.module.css';

export default function CollapsibleCodeBlock({ children, title, defaultOpen = false, ...props }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const toggleOpen = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={styles.collapsibleCodeBlock}>
      <div
        className={clsx(styles.header, { [styles.headerOpen]: isOpen })}
        onClick={toggleOpen}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && toggleOpen()}
      >
        <span className={styles.title}>{title || 'Code Example'}</span>
        <span className={clsx(styles.chevron, { [styles.chevronOpen]: isOpen })}>
          ▼
        </span>
      </div>
      <div
        className={clsx(styles.content, { [styles.contentOpen]: isOpen })}
        style={{
          maxHeight: isOpen ? '1000px' : '0',
          opacity: isOpen ? '1' : '0'
        }}
      >
        <CodeBlock {...props}>
          {children}
        </CodeBlock>
      </div>
    </div>
  );
}