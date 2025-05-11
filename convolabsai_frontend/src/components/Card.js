import React from 'react';


const Card = ({ children, className }) => {
  return (
    <div className={`rounded-lg shadow-md border ${className}`}>{children}</div>
  );
};


const CardHeader = ({ children, className }) => {
  return (
    <div className={`p-4 border-b ${className}`}>{children}</div>
  );
};


const CardTitle = ({ children, className }) => {
  return (
    <h2 className={`text-lg font-semibold ${className}`}>{children}</h2>
  );
};


const CardContent = ({ children, className }) => {
  return (
    <div className={`p-4 ${className}`}>{children}</div>
  );
};

export { Card, CardHeader, CardTitle, CardContent };
