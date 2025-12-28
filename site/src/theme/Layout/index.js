import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ReadingProgress from '@site/src/components/ReadingProgress';

export default function Layout(props) {
  return (
    <>
      <ReadingProgress />
      <OriginalLayout {...props} />
    </>
  );
}