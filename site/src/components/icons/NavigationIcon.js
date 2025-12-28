import React, { memo } from 'react';

const NavigationIcon = memo(({ size = 'md', className, ...props }) => {
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
      <path d="M12 2L15.09 8.26L22 9L17 14L18.18 21L12 17.77L5.82 21L7 14L2 9L8.91 8.26L12 2Z" fill="#3399CC" />
    </svg>
  );
});

export default NavigationIcon;