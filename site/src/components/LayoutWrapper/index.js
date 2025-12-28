import React from 'react';
import Layout from '@theme/Layout';
import BackToTop from '@site/src/components/BackToTop';

function LayoutWrapper(props) {
  return (
    <>
      <Layout {...props}>
        {props.children}
        <BackToTop />
      </Layout>
    </>
  );
}

export default LayoutWrapper;