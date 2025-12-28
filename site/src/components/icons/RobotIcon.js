import React, { memo } from 'react';

const RobotIcon = memo(({ size = 'md', className, ...props }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const sizeClass = sizes[size] || sizes.md;
  const combinedClassName = `${sizeClass} ${className || ''}`;

  return (
    <svg
      className={combinedClassName}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <rect x="6" y="4" width="12" height="16" rx="2" fill="#444" />
      <circle cx="9" cy="9" r="1.5" fill="#fff" />
      <circle cx="15" cy="9" r="1.5" fill="#fff" />
      <rect x="8" y="13" width="8" height="3" rx="1" fill="#666" />
    </svg>
  );
});

export default RobotIcon;