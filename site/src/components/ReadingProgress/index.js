import React, { useState, useEffect } from 'react';

import styles from './styles.module.css';

export default function ReadingProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY;
      const windowHeight = window.innerHeight;
      const docHeight = Math.max(
        document.body.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.clientHeight,
        document.documentElement.scrollHeight,
        document.documentElement.offsetHeight
      );

      const totalScrollable = docHeight - windowHeight;
      const scrollPercentage = totalScrollable > 0 ? (scrollTop / totalScrollable) * 100 : 0;

      setProgress(Math.min(scrollPercentage, 100));
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial calculation

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  // Only show on pages with significant content
  if (progress < 5) {
    return null;
  }

  return (
    <div className={styles.progressBarContainer}>
      <div
        className={styles.progressBar}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}