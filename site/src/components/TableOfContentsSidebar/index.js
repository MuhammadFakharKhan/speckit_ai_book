import React, { useState, useEffect } from 'react';
import { useActiveDocContext } from '@docusaurus/plugin-content-docs/client';
import Link from '@docusaurus/Link';

import styles from './styles.module.css';

export default function TableOfContentsSidebar() {
  const { activeDoc } = useActiveDocContext();
  const [activeId, setActiveId] = useState('');

  useEffect(() => {
    const handleScroll = () => {
      const headers = Array.from(document.querySelectorAll('h1, h2, h3, h4'))
        .filter(header => header.id)
        .map(header => ({
          id: header.id,
          offsetTop: header.offsetTop,
        }))
        .sort((a, b) => a.offsetTop - b.offsetTop);

      const scrollPosition = window.scrollY + 100; // Add offset for navbar

      for (let i = headers.length - 1; i >= 0; i--) {
        if (scrollPosition >= headers[i].offsetTop) {
          setActiveId(headers[i].id);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial call

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  if (!activeDoc || !activeDoc.toc) {
    return null;
  }

  const toc = activeDoc.toc;

  if (toc.length <= 1) {
    return null; // Don't show if there's only one item
  }

  return (
    <div className={styles.tocSidebar}>
      <h3 className={styles.tocTitle}>On this page</h3>
      <ul className={styles.tocList}>
        {toc.map((item) => (
          <li
            key={item.id}
            className={`${styles.tocItem} ${
              item.id === activeId ? styles.tocItemActive : ''
            }`}
            style={{ paddingLeft: `${item.level - 2}rem` }}
          >
            <Link
              to={`#${item.id}`}
              className={`${styles.tocLink} ${
                item.id === activeId ? styles.tocLinkActive : ''
              }`}
              onClick={(e) => {
                e.preventDefault();
                const element = document.getElementById(item.id);
                if (element) {
                  element.scrollIntoView({ behavior: 'smooth' });
                  window.scrollTo({ top: window.scrollY - 80 }); // Adjust for fixed header
                }
              }}
            >
              {item.value}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}