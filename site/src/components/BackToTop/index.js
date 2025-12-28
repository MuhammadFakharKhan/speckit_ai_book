import React, { useState, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useLocation } from '@docusaurus/router';

const BackToTop = () => {
  const [showButton, setShowButton] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      // Show button when page is scrolled down more than 300px
      if (window.scrollY > 300) {
        setShowButton(true);
      } else {
        setShowButton(false);
      }
    };

    // Add scroll event listener
    window.addEventListener('scroll', handleScroll);

    // Clean up event listener on component unmount
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleClick = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  };

  // Only show the button on docs pages, not on the homepage
  if (pathname === '/' || pathname === '/docs') {
    return null;
  }

  return (
    <button
      className={`back-to-top ${showButton ? 'visible' : ''}`}
      onClick={handleClick}
      aria-label="Back to top"
      title="Back to top"
    />
  );
};

export default BackToTop;