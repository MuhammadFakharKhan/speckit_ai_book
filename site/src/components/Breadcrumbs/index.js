import React from 'react';
import { useLocation } from '@docusaurus/router';
import { useActiveDocContext } from '@docusaurus/plugin-content-docs/client';
import Link from '@docusaurus/Link';
import clsx from 'clsx';

import styles from './styles.module.css';

function BreadcrumbItem({ href, isLast, children }) {
  if (isLast) {
    return (
      <li class={clsx(styles.breadcrumbItem, styles.breadcrumbItemActive)}>
        {children}
      </li>
    );
  }
  return (
    <li class={styles.breadcrumbItem}>
      {href ? <Link href={href}>{children}</Link> : children}
    </li>
  );
}

export default function Breadcrumbs() {
  const location = useLocation();
  const { activeDoc } = useActiveDocContext();

  if (!activeDoc) {
    return null;
  }

  // Generate breadcrumbs based on the document path
  const { slug, sidebar } = activeDoc;
  if (!slug || slug === '/docs/intro') {
    return null; // Don't show breadcrumbs on the intro page
  }

  // Split the slug to create breadcrumb segments
  const segments = slug.split('/').filter(segment => segment !== '');

  // Create breadcrumb items
  const breadcrumbs = segments.map((segment, index) => {
    const isLast = index === segments.length - 1;
    const href = isLast ? null : `/${segments.slice(0, index + 1).join('/')}`;

    // Convert kebab-case to Title Case
    const title = segment
      .split('-')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    return (
      <BreadcrumbItem key={index} href={href} isLast={isLast}>
        {title}
      </BreadcrumbItem>
    );
  });

  return (
    <nav className={styles.breadcrumbNav} aria-label="Breadcrumb">
      <ol className={styles.breadcrumbList}>
        <BreadcrumbItem href="/docs/intro">Home</BreadcrumbItem>
        {breadcrumbs}
      </ol>
    </nav>
  );
}