import React from 'react';
import clsx from 'clsx';
import OriginalDocItemLayout from '@theme-original/DocItem/Layout';
import Breadcrumbs from '@site/src/components/Breadcrumbs';
import TableOfContentsSidebar from '@site/src/components/TableOfContentsSidebar';

export default function DocItemLayout(props) {
  const { content: DocContent } = props;
  const hasToc = DocContent && DocContent.metadata && DocContent.metadata.toc && DocContent.metadata.toc.length > 0;

  return (
    <>
      <div className="container margin-vert--lg">
        <Breadcrumbs />
        <div className="row">
          <div className={clsx('col', { 'col--8': hasToc })}>
            <OriginalDocItemLayout {...props} />
          </div>
          {hasToc && (
            <div className="col col--2">
              <TableOfContentsSidebar />
            </div>
          )}
        </div>
      </div>
    </>
  );
}