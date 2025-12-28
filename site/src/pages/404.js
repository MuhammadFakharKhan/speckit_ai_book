import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import styles from './404.module.css';

export default function NotFound() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout title={`Page Not Found | ${siteConfig.title}`} description="Page not found">
      <div className={styles.container}>
        <div className={styles.content}>
          <div className={styles.illustration}>
            <div className={styles.robotIcon}>🤖</div>
          </div>

          <h1 className={styles.title}>404</h1>
          <p className={styles.subtitle}>Page Not Found</p>

          <p className={styles.message}>
            We couldn't find the page you're looking for. It might have been moved,
            renamed, or doesn't exist in the ROS 2 documentation.
          </p>

          <div className={styles.actions}>
            <Link className={styles.primaryButton} to="/">
              Go to Homepage
            </Link>
            <Link className={styles.secondaryButton} to="/docs/intro">
              Browse Documentation
            </Link>
          </div>

          <div className={styles.searchSection}>
            <p>Try searching for what you need:</p>
            <div className="search-bar">
              <input
                type="text"
                placeholder="Search documentation..."
                className={styles.searchInput}
                onFocus={(e) => {
                  // If the theme provides a search modal, we can trigger it here
                  // For now, we'll just focus the input
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}